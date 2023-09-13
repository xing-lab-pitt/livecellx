import fiftyone as fo
import fiftyone.zoo as foz


if __name__ == "__main__":
    dataset = foz.load_zoo_dataset("quickstart")
    session = fo.launch_app(dataset, desktop=False, port=5156)
    session.wait()
