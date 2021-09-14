import uuid


class SimpleClipStorage:
    def __init__(self):
        self.store = {}

    def load(self, clip_id):
        if clip_id in self.store:
            return self.store[clip_id]
        raise Exception(f"clip_id {clip_id} not found")

    def save(self, audio):
        clip_id = uuid.uuid4()
        self.store[clip_id] = audio
        return clip_id
