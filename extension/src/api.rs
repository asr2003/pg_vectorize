use crate::chat::ops::{call_chat, call_chat_completions};
use crate::chat::types::RenderedPrompt;
use crate::guc::get_guc_configs;
use crate::search::{self, init_table};
use crate::transformers::generic::env_interpolate_string;
use crate::transformers::transform;
use crate::types;
use crate::util::*;

use anyhow::Result;
use pgrx::prelude::*;
use pgrx::PgOid;
use sqlx::Row;
use vectorize_core::types::Model;

#[allow(clippy::too_many_arguments)]
#[pg_extern]
async fn table(
    table: &str,
    columns: Vec<String>,
    job_name: &str,
    primary_key: &str,
    schema: default!(&str, "'public'"),
    update_col: default!(String, "'last_updated_at'"),
    index_dist_type: default!(types::IndexDist, "'pgv_hnsw_cosine'"),
    transformer: default!(&str, "'sentence-transformers/all-MiniLM-L6-v2'"),
    chunk_size: default!(Option<i32>, "NULL"),
    chunk_overlap: default!(Option<i32>, "NULL"),
    // search_alg is now deprecated
    search_alg: default!(types::SimilarityAlg, "'pgv_cosine_similarity'"),
    table_method: default!(types::TableMethod, "'join'"),
    // cron-like for a cron based update model, or 'realtime' for a trigger-based
    schedule: default!(&str, "'* * * * *'"),
) -> Result<String> {
    let processed_table = if let Some(chunk_size) = chunk_size {
        let chunked_table_name = format!("{}_chunked", table);
        chunk_table(
            table,
            columns.clone(),
            chunk_size,
            chunk_overlap.unwrap_or(200),
            &chunked_table_name,
            schema,
        )
        .await?;
        chunked_table_name
    } else {
        table.to_string()
    };

    let model = Model::new(transformer)?;
    init_table(
        job_name,
        schema,
        &processed_table,
        columns,
        primary_key,
        Some(update_col),
        index_dist_type.into(),
        &model,
        table_method.into(),
        schedule,
    )
}

/// Create a chunked table with the necessary schema
pub fn create_chunked_table(
    table_name: &str,
    columns: Vec<String>,
    schema: &str,
) -> Result<()> {
    let query = format!(
        "CREATE TABLE {schema}.{table_name} (
            id SERIAL PRIMARY KEY,
            original_id INTEGER,
            chunk TEXT
        );"
    );
    Spi::run(&query)?;
    Ok(())
}

/// Insert a chunk into the chunked table
pub fn insert_chunk_into_table(
    table_name: &str,
    chunk: String,
    original_id: i32,
    schema: &str,
) -> Result<()> {
    let query = format!(
        "INSERT INTO {schema}.{table_name} (original_id, chunk) VALUES ($1, $2);"
    );
    Spi::run_with_args(
        &query,
        Some(vec![
            (PgOid::from(23), Some(original_id.into_datum().unwrap())),
            (PgOid::from(25), Some(chunk.into_datum().unwrap())),
        ]),
    )?;
    Ok(())
}

/// Utility function to chunk the rows of a table and store them in a new table
#[pg_extern]
async fn chunk_table(
    input_table: &str,
    columns: Vec<String>,
    chunk_size: default!(i32, 1000),
    chunk_overlap: default!(i32, 200),
    output_table: &str,
    schema: default!(&str, "'public'"),
) -> Result<String> {
    let conn = get_pg_conn().await?;

    let rows = fetch_table_rows(&conn, input_table, columns.clone(), schema).await?;
    create_chunked_table(output_table, columns.clone(), schema)?;

    for row in rows {
        for col in &columns {
            if let Some(text) = row.get(col) {
                let chunks =
                    chunking::chunk_text(text, chunk_size as usize, chunk_overlap as usize);
                // Insert each chunk as a new row in the output table
                for chunk in chunks {
                    insert_chunk_into_table(
                        output_table,
                        chunk,
                        row.get("primary_key").unwrap(),
                        schema,
                    )?;
                }
            }
        }
    }

    Ok(format!(
        "Data from {} successfully chunked into {}",
        input_table, output_table
    ))
}

#[pg_extern]
fn search(
    job_name: String,
    query: String,
    api_key: default!(Option<String>, "NULL"),
    return_columns: default!(Vec<String>, "ARRAY['*']::text[]"),
    num_results: default!(i32, 10),
    where_sql: default!(Option<String>, "NULL"),
) -> Result<TableIterator<'static, (name!(search_results, pgrx::JsonB),)>> {
    let search_results = search::search(
        &job_name,
        &query,
        api_key,
        return_columns,
        num_results,
        where_sql,
    )?;
    Ok(TableIterator::new(search_results.into_iter().map(|r| (r,))))
}

#[pg_extern]
fn transform_embeddings(
    input: &str,
    model_name: default!(String, "'sentence-transformers/all-MiniLM-L6-v2'"),
    api_key: default!(Option<String>, "NULL"),
) -> Result<Vec<f64>> {
    let model = Model::new(&model_name)?;
    Ok(transform(input, &model, api_key).remove(0))
}

#[pg_extern]
fn encode(
    input: &str,
    model: default!(String, "'sentence-transformers/all-MiniLM-L6-v2'"),
    api_key: default!(Option<String>, "NULL"),
) -> Result<Vec<f64>> {
    let model = Model::new(&model)?;
    Ok(transform(input, &model, api_key).remove(0))
}

#[allow(clippy::too_many_arguments)]
#[pg_extern]
fn init_rag(
    agent_name: &str,
    table_name: &str,
    unique_record_id: &str,
    // column that have data we want to be able to chat with
    column: &str,
    schema: default!(&str, "'public'"),
    index_dist_type: default!(types::IndexDist, "'pgv_hnsw_cosine'"),
    // transformer model to use in vector-search
    transformer: default!(&str, "'sentence-transformers/all-MiniLM-L6-v2'"),
    table_method: default!(types::TableMethod, "'join'"),
    schedule: default!(&str, "'* * * * *'"),
) -> Result<String> {
    // chat only supports single columns transform
    let columns = vec![column.to_string()];
    let transformer_model = Model::new(transformer)?;
    init_table(
        agent_name,
        schema,
        table_name,
        columns,
        unique_record_id,
        None,
        index_dist_type.into(),
        &transformer_model,
        table_method.into(),
        schedule,
    )
}

/// creates a table indexed with embeddings for chat completion workloads
#[pg_extern]
fn rag(
    agent_name: &str,
    query: &str,
    chat_model: default!(String, "'tembo/meta-llama/Meta-Llama-3-8B-Instruct'"),
    // points to the type of prompt template to use
    task: default!(String, "'question_answer'"),
    api_key: default!(Option<String>, "NULL"),
    // number of records to include in the context
    num_context: default!(i32, 2),
    // truncates context to fit the model's context window
    force_trim: default!(bool, false),
) -> Result<TableIterator<'static, (name!(chat_results, pgrx::JsonB),)>> {
    let model = Model::new(&chat_model)?;
    let resp = call_chat(
        agent_name,
        query,
        &model,
        &task,
        api_key,
        num_context,
        force_trim,
    )?;
    let iter = vec![(pgrx::JsonB(serde_json::to_value(resp)?),)];
    Ok(TableIterator::new(iter))
}

#[pg_extern]
fn generate(
    input: &str,
    model: default!(String, "'tembo/meta-llama/Meta-Llama-3-8B-Instruct'"),
    api_key: default!(Option<String>, "NULL"),
) -> Result<String> {
    let model = Model::new(&model)?;
    let prompt = RenderedPrompt {
        sys_rendered: "".to_string(),
        user_rendered: input.to_string(),
    };
    let mut guc_configs = get_guc_configs(&model.source);
    if let Some(api_key) = api_key {
        guc_configs.api_key = Some(api_key);
    }
    call_chat_completions(prompt, &model, &guc_configs)
}

#[pg_extern]
fn env_interpolate_guc(guc_name: &str) -> Result<String> {
    let g: String = Spi::get_one_with_args(
        "SELECT current_setting($1)",
        vec![(PgBuiltInOids::TEXTOID.oid(), guc_name.into_datum())],
    )?
    .unwrap_or_else(|| panic!("no value set for guc: {guc_name}"));
    env_interpolate_string(&g)
}
