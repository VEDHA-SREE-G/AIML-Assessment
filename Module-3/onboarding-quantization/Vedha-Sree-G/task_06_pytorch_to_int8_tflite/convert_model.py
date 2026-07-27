import os
import sys
import logging
import numpy as np
import torch
import tensorflow as tf
import onnx

from model_definition import SimpleCNN


LOG_FILE = "conversion_log.txt"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


CHECKPOINT = "model.pth"
ONNX_FILE = "model.onnx"
SAVED_MODEL_DIR = "saved_model"
TFLITE_FILE = "model_int8.tflite"

INPUT_SHAPE = (1, 1, 28, 28)
CALIB_DIR = "calib"

def log(msg):
    print(msg)
    logging.info(msg)

def load_model():

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint missing: {CHECKPOINT}"
        )

    model = SimpleCNN()

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    log("Model loaded successfully")

    return model

def validate_calibration():

    if not os.path.exists(CALIB_DIR):
        raise FileNotFoundError(
            "Calibration directory missing"
        )

    files = sorted([
        os.path.join(CALIB_DIR, f)
        for f in os.listdir(CALIB_DIR)
        if f.endswith(".npy")
    ])

    if len(files) == 0:
        raise RuntimeError(
            "Calibration directory empty"
        )

    samples = []

    for file in files:

        try:
            arr = np.load(file)

        except Exception as e:
            raise RuntimeError(
                f"Invalid file {file}: {e}"
            )

        if arr.shape != (1, 28, 28):
            raise ValueError(
                f"Bad shape {arr.shape} in {file}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(
                f"NaN/Inf detected in {file}"
            )

        if arr.dtype not in (
            np.float32,
            np.float64
        ):
            raise ValueError(
                f"Unexpected dtype {arr.dtype}"
            )

        samples.append(arr.astype(np.float32))

    log(
        f"Validated {len(samples)} calibration samples"
    )

    return samples

def export_onnx(model):

    dummy = torch.randn(INPUT_SHAPE)

    torch.onnx.export(
    model,
    dummy,
    ONNX_FILE,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False   # ADD THIS
)

    onnx_model = onnx.load(ONNX_FILE)

    onnx.checker.check_model(
        onnx_model
    )

    log("ONNX export verified")

from onnx_tf.backend import prepare


def onnx_to_savedmodel():

    import shutil

    if os.path.exists(SAVED_MODEL_DIR):
        shutil.rmtree(SAVED_MODEL_DIR)

    onnx_model = onnx.load(ONNX_FILE)

    tf_rep = prepare(onnx_model)

    tf_rep.export_graph(SAVED_MODEL_DIR)

    log("SavedModel exported successfully")

def representative_dataset(samples):

    for arr in samples:

        arr = np.expand_dims(
            arr,
            axis=0
        )

        arr = arr.astype(np.float32)

        yield [arr]

def convert_int8(samples):

    converter = (
        tf.lite.TFLiteConverter
        .from_saved_model(
            SAVED_MODEL_DIR
        )
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        lambda:
        representative_dataset(samples)
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(TFLITE_FILE, "wb") as f:
        f.write(tflite_model)

    log("INT8 TFLite created")

def verify_tflite():

    interpreter = tf.lite.Interpreter(
        model_path=TFLITE_FILE
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    inp = input_details[0]
    out = output_details[0]

    print("\n===== VERIFICATION =====")

    print(
        "Input dtype:",
        inp["dtype"]
    )

    print(
        "Output dtype:",
        out["dtype"]
    )

    print(
        "Input quantization:",
        inp["quantization"]
    )

    print(
        "Output quantization:",
        out["quantization"]
    )

    if inp["dtype"] != np.int8:
        raise RuntimeError(
            "Floating-point input detected"
        )

    if out["dtype"] != np.int8:
        raise RuntimeError(
            "Floating-point output detected"
        )

    shape = inp["shape"]

    dummy = np.zeros(
        shape,
        dtype=np.int8
    )

    interpreter.set_tensor(
        inp["index"],
        dummy
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        out["index"]
    )

    print(
        "Inference output:"
    )

    print(output)

    size_kib = (
        os.path.getsize(TFLITE_FILE)
        / 1024
    )

    print(
        f"Model size: {size_kib:.2f} KiB"
    )

    log("Verification complete")

def main():

    try:

        model = load_model()

        samples = validate_calibration()

        export_onnx(model)

        onnx_to_savedmodel()

        convert_int8(samples)

        verify_tflite()

        log("SUCCESS")

    except Exception as e:

        log(f"FAILED: {e}")

        raise


if __name__ == "__main__":
    main()