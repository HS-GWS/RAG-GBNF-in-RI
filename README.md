# RAG and GBNF application for creating Argument Maps from Regesta Imperii Data

In this repository you find everything you need to locally run a basic streamlit app to create Argument Maps with a llama.cpp model from Regesta Imperii files. You may proced as follows:

1. Install all packages that are stated in the requirements.txt
2. Produce a vectorstore from one of the CSV-Files from the [Regesta Imperii repository https://gitlab.rlp.net/adwmainz/regesta-imperii/lab/regesta-imperii-data] with the CreateVectorstore.py
   PLEASE NOTE: This will take a while!
4. Run the App from the Commandline with either:
   ```streamlit run ri_argument_mapping.py```
   or
   ```python -m streamlit run ri_argument_mapping.py```

This is a work in progress which is highly dependent on packeges that are undergoing rapid changes right now(Status September 2024).
