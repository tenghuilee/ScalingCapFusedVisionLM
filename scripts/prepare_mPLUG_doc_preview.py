# a small web
import bottle
import json
import http
from bottle import Bottle, run, request, response, static_file, abort

class ItemTaker:
    def __init__(self):
        # self.sub_files = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_{rank:02d}_of_09_nougat_clean.jsonl"
        self.sub_files = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math_droped.jsonl"
        self.out_file = "../datasets/temp_manual_edit.jsonl"
        self.reject_files = "../datasets/temp_reject.jsonl"
        self.ostream = open(self.out_file, 'a')
        self.reject_stream = open(self.reject_files, 'a')

        self.current_item = None
        self.default_iter = iter(self.iter())

        self.passed_images = set()
        # self._reset()
    
    def _reset(self):
        self.passed_images.clear()
        with open(self.out_file, 'r') as f:
            for line in f:
                self.passed_images.add(json.loads(line)["image"])
        with open(self.reject_files, 'r') as f:
            for line in f:
                self.passed_images.add(json.loads(line)["image"])

    def __del__(self):
        if hasattr(self, 'ostream'):
            self.ostream.close()
        if hasattr(self, 'reject_stream'):
            self.reject_stream.close()
    
    def write(self):
        if self.current_item is None:
            return
        # self.ostream.write(json.dumps(self.current_item) + "\n")
    
    def reject(self):
        if self.current_item is None:
            return
        # self.reject_stream.write(json.dumps(self.current_item) + "\n")
    
    def iter(self):
        for r in range(9):
            with open(self.sub_files.format(rank=r), 'r') as f:
                for idx, line in enumerate(f):
                    item = json.loads(line)
                    if item["image"] in self.passed_images:
                        continue
                    yield item
    
    def next(self):
        self.current_item = next(self.default_iter)
        return self.current_item

    def reset(self):
        self.current_item = None
        self.default_iter = iter(self.iter())
        self._reset()


item_taker = ItemTaker()

app = Bottle()

@app.route('/')
def home():
    return static_file('mPLUG_doc_preview_index.html', root='.')

@app.get('/api/mark/<status>')
def mark_status(status: str):
    current = item_taker.current_item
    if status == 'accept':
        item_taker.write()
    elif status == 'reject':
        item_taker.reject()
    elif status == 'reset':
        item_taker.reset()
        return item_taker.next()
    else:
        abort(http.HTTPStatus.BAD_REQUEST, "Invalid status")
        # return 500
        return {"id": None}
    return {"id": current["image"]}

@app.get("/static/<image_name:path>")
def get_image(image_name: str):
    return static_file(image_name, root='../datasets/soft_link_image_collection')

@app.get('/api/get_next')
def iter_next():
    return item_taker.next()

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=8080, debug=True)

