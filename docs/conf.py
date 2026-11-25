project = 'SIMBA CGM line-emission pipeline'
author = 'Tang Jia Heak'

extensions = ['myst_parser']
root_doc = 'index'
exclude_patterns = ['_build']
nitpicky = True
myst_heading_anchors = 3

html_theme = 'furo'
html_title = 'SIMBA gas emission'
html_theme_options = {
    'source_repository': 'https://github.com/JiaHeak/SIMBA-gas-emission/',
    'source_branch': 'develop',
    'source_directory': 'docs/',
}
