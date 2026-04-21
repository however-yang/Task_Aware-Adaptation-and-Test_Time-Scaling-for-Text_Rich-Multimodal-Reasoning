# Evaluation Report

| Metric | Value |
| --- | ---: |
| docvqa | 0.7609 |
| chartqa | 0.6807 |
| overall | 0.7123 |
| num_predictions | 3170 |

## Error Counts

| Error Type | Count |
| --- | ---: |
| text_reading_failure | 609 |
| correct | 1948 |
| chart_reasoning_failure | 613 |

## by_dataset

| Key | Count | Mean Score |
| --- | ---: | ---: |
| docvqa | 1250 | 0.7609 |
| chartqa | 1920 | 0.6807 |

## by_task_type

| Key | Count | Mean Score |
| --- | ---: | ---: |
| document_qa | 1250 | 0.7609 |
| chart_qa | 1920 | 0.6807 |

## by_answer_type

| Key | Count | Mean Score |
| --- | ---: | ---: |
| open_text | 1727 | 0.7464 |
| numeric | 1443 | 0.6715 |

## by_split

| Key | Count | Mean Score |
| --- | ---: | ---: |
| validation | 3170 | 0.7123 |

## by_error_type

| Key | Count | Mean Score |
| --- | ---: | ---: |
| text_reading_failure | 609 | 0.5092 |
| correct | 1948 | 1.0000 |
| chart_reasoning_failure | 613 | 0.0000 |
