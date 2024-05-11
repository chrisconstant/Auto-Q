import pandas as pd
import torch
from llm2vec import LLM2Vec
from sentence_transformers import SentenceTransformer
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser
import torch

sets = [
    {
        "train": "train.csv",
        "dest": "dev.csv",
        "mode": "val",
    },
    {
        "train": "train.csv",
        "dest": "test.csv",
        "mode": "test",
    },
]

models = [
    {
        "mdl": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
        "mode": "Mistral_7B_Instruct_v2_mntp_unsup_simcse",
        "type": "LLM2vec",
    },
    {
        "mdl": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
        "mode": "Mistral_7B_Instruct_v2_mntp_supervised",
        "type": "LLM2vec",
    },
    {"mdl": "all-mpnet-base-v2", "mode": "all_mpnet_base_v2", "type": "ST", "query": ""},
    {"mdl": "intfloat/e5-small-v2", "mode": "e5_small_v2", "type": "ST", "query": "query: "},
    {"mdl": "intfloat/e5-large-v2", "mode": "e5_large_v2", "type": "ST", "query": "query: "},
    # {
    #    "mdl": "jspringer/echo-mistral-7b-instruct-lasttoken",
    #    "mode": "echo_mistral_7b_instruct_lasttoken",
    #    "type": "Echo",
    # },
]

for db in sets:
    for mdl in models:
        if mdl["type"] == "LLM2vec":
            l2v = LLM2Vec.from_pretrained(
                mdl["mdl"],
                peft_model_name_or_path=mdl["peft_model_name_or_path"],
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )

            prefixmatch = mdl["mode"] + "_" + db["mode"]

            df = pd.read_csv(db["train"])
            documents = list(df["workordertext"])
            d_reps = l2v.encode(documents)

            df = pd.read_csv(db["dest"])
            querys = list(df["workordertext"])
            instruction = "Given a workorder description, retrieve similar workorder:"
            queries = [instruction + " " + item for item in querys]
            q_reps = l2v.encode(queries)

            # passing the results
            q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
            d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
            cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

            top_values, top_indices = torch.topk(cos_sim, k=3, dim=1)
            top_values_list = top_values.tolist()
            top_indices_list = top_indices.tolist()

            # Convert lists to pandas DataFrames
            df_values = pd.DataFrame(
                top_values_list, columns=["Top Value 1", "Top Value 2", "Top Value 3"]
            )
            df_indices = pd.DataFrame(
                top_indices_list, columns=["Top Index 1", "Top Index 2", "Top Index 3"]
            )

            # Concatenate DataFrames horizontally (along columns)
            df_merged = pd.concat([df_values, df_indices], axis=1)
            df_merged.to_csv(prefixmatch + ".csv", index=False)
        elif mdl["type"] == "ST":
            sentence_model = SentenceTransformer(mdl["mdl"])
            prefixmatch = mdl["mode"] + "_" + db["mode"]

            df = pd.read_csv(db["train"])
            documents = list(df["workordertext"])

            df = pd.read_csv(db["dest"])
            querys = list(df["workordertext"])

            if len(mdl['query']) > 0:
                documents = [mdl['query'] + item for item in documents]
                querys = [mdl['query'] + item for item in querys]


            train_reps = sentence_model.encode(documents)
            test_reps = sentence_model.encode(querys)

            # Normalize representations
            train_reps_norm = torch.nn.functional.normalize(
                torch.tensor(train_reps), p=2, dim=1
            )
            test_reps_norm = torch.nn.functional.normalize(
                torch.tensor(test_reps), p=2, dim=1
            )

            # Calculate cosine similarity
            cos_sim = torch.mm(test_reps_norm, train_reps_norm.transpose(0, 1))

            # Get top k indices and values
            top_values, top_indices = torch.topk(cos_sim, k=3, dim=1)
            top_values_list = top_values.tolist()
            top_indices_list = top_indices.tolist()

            # Convert lists to pandas DataFrames
            df_values = pd.DataFrame(
                top_values_list, columns=[f"Top Value {i+1}" for i in range(3)]
            )
            df_indices = pd.DataFrame(
                top_indices_list, columns=[f"Top Index {i+1}" for i in range(3)]
            )

            # Concatenate DataFrames horizontally (along columns)
            df_merged = pd.concat([df_values, df_indices], axis=1)
            df_merged.to_csv(prefixmatch + ".csv", index=False)
        elif mdl["type"] == "Echo":
            templates = {
                "query": "<s>Instruct:{!%%prompt%%,}\nQuery:{!%%text%%}\nQuery again:{%%text%%}{</s>}",
                "document": "<s>Document:{!%%text%%}\nDocument again:{%%text%%}{</s>}",
            }

            # Create the model
            path_to_model = mdl["mdl"]
            model = EchoEmbeddingsMistral.from_pretrained(path_to_model)
            model = model.eval()

            # Create the parser
            parser = EchoParser(path_to_model, templates, max_length=300)

            # Create the pooling: strategy can either be mean or last
            pooling = EchoPooling(strategy="last")

            df = pd.read_csv(db["train"])
            documents = list(df["workordertext"])

            df = pd.read_csv(db["dest"])
            queries = list(df["workordertext"])

            # specify the prompt, queries, and documents
            prompt = "Find similar workorder:"

            query_variables = [{"prompt": prompt, "text": q} for q in queries]
            document_variables = [{"text": d} for d in documents]

            query_tagged = [("query", q) for q in query_variables]
            document_tagged = [("document", d) for d in document_variables]

            # Get the tokenized embeddings
            top_values_list = []
            top_indices_list = []

            for query in query_tagged:
                with torch.no_grad():
                    query_embeddings = pooling(model(parser([query])))[
                        "sentence_embedding"
                    ]

                sim_distance = []

                for document in document_tagged:
                    # Document
                    with torch.no_grad():
                        document_embeddings = pooling(model(parser([document])))[
                            "sentence_embedding"
                        ]

                    # compute the cosine similarity
                    sim = lambda x, y: torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

                    sim_distance.append(
                        sim(query_embeddings[0], document_embeddings[0])
                    )

                # now discover the
                # Enumerate the list to preserve the indices
                indexed_distances = list(enumerate(sim_distance))

                # Sort the list based on cosine values in descending order
                sorted_distances = sorted(
                    indexed_distances, key=lambda x: x[1], reverse=True
                )

                # Get the top 3 values and their indices
                top_3 = sorted_distances[:3]
                top_indices_list.append([top_3[0][0], top_3[1][0], top_3[2][0]])
                top_values_list.append([top_3[0][1], top_3[1][1], top_3[2][1]])

            # Convert lists to pandas DataFrames
            df_values = pd.DataFrame(
                top_values_list, columns=[f"Top Value {i+1}" for i in range(3)]
            )
            df_indices = pd.DataFrame(
                top_indices_list, columns=[f"Top Index {i+1}" for i in range(3)]
            )

            # Concatenate DataFrames horizontally (along columns)
            df_merged = pd.concat([df_values, df_indices], axis=1)
            df_merged.to_csv(prefixmatch + ".csv", index=False)
