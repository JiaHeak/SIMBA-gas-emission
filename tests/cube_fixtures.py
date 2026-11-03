"""Small prepared-table stub for controller tests that mock emission physics."""


def mock_emissivity_table(table_type, table_filename, line_labels):
    """Supply only the metadata needed by controller routing tests.

    Returns: emissivity_table (dict): fake table metadata without a file read.
    """
    return {'table_type': table_type, 'table_filename': str(table_filename),
            'line_labels': line_labels, 'data': {'axes': {}, 'line_labels': line_labels}}
