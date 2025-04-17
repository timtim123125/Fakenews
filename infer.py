from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from transformers import AutoTokenizer

type_list = ["Problem", "Request", "Change"]
queue_list = [
    "Technical",
    "Billing",
    "Product",
    "Infra",
]


def softmax(x: NDArray[Any]) -> NDArray[Any]:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def run_inference(
    text: str,
    model_path: str,
    tokenizer_name: str = "bert-base-uncased",
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    inputs = tokenizer(
        text.lower(),
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="np",
    )

    session = ort.InferenceSession(model_path)

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    outputs = session.run(None, ort_inputs)

    logits_head1 = outputs[0]  # shape: (batch_size, num_classes_head1)
    logits_head2 = outputs[1]  # shape: (batch_size, num_classes_head2)

    # Apply softmax to convert logits to probabilities (optional; for interpretation)
    probs_head1 = softmax(logits_head1)
    probs_head2 = softmax(logits_head2)

    # Get the predicted class indexes.
    pred_class_head1 = np.argmax(probs_head1, axis=-1)[0]
    pred_class_head2 = np.argmax(probs_head2, axis=-1)[0]

    # Convert probabilities to list of dictionaries
    type_probs = [
        {"name": name, "prob": float(prob)}
        for name, prob in zip(type_list, np.round(probs_head1[0], 2))
    ]

    queue_probs = [
        {"name": name, "prob": float(prob)}
        for name, prob in zip(queue_list, np.round(probs_head2[0], 2))
    ]

    return (
        type_list[pred_class_head1],
        queue_list[pred_class_head2],
        type_probs,
        queue_probs,
    )
