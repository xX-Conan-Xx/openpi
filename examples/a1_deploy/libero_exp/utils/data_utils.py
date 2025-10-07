import torch
from hydra.utils import to_absolute_path
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from transformers import AutoModel, AutoTokenizer, logging
from libero.lifelong.utils import safe_device


def get_task_embs(cfg, descriptions, embedding_model_path=None):
    logging.set_verbosity_error()

    if cfg.data.task_embedding_format == "one-hot":
        # offset defaults to 1, if we have pretrained another model, this offset
        # starts from the pretrained number of tasks + 1
        offset = cfg.task_embedding_one_hot_offset
        descriptions = [f"Task {i+offset}" for i in range(len(descriptions))]

    if cfg.data.task_embedding_format == "bert" or cfg.data.task_embedding_format == "one-hot":
        if embedding_model_path != None:
            tz = AutoTokenizer.from_pretrained(embedding_model_path)
            model = AutoModel.from_pretrained(embedding_model_path)
        else:
            tz = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=to_absolute_path("./bert"))
            model = AutoModel.from_pretrained("bert-base-cased", cache_dir=to_absolute_path("./bert"))
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif cfg.data.task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModel.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif cfg.data.task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif cfg.data.task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModel.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()

    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]

    return task_embs


def raw_obs_to_tensor_obs(obs, task_emb, cfg, device=None):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "obs": {},
        "task_emb": task_emb.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    if device == None:
        data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.train.device))
    elif device == 'cpu':
        data = TensorUtils.map_tensor(data, lambda x: safe_device(x))
    
    return data


