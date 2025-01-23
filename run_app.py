import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Run the Streamlit app
if __name__ == '__main__':
    import streamlit.web.cli as stcli
    import sys
    
    sys.argv = ["streamlit", "run", "app/main.py"]
    sys.exit(stcli.main())
