import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from scanner import scan_document, rotate_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filter_type = request.form.get('filter', 'bw')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = 'scanned_' + filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    file.save(input_path)

    try:
        scan_document(input_path, output_path, filter_type)
        return jsonify({
            'original': '/' + input_path.replace('\\', '/'),
            'scanned': '/' + output_path.replace('\\', '/'),
            'output_filename': output_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/refilter', methods=['POST'])
def refilter():
    data = request.get_json()
    output_filename = data.get('filename')
    filter_type = data.get('filter', 'bw')
    input_filename = output_filename.replace('scanned_', '', 1)

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    try:
        scan_document(input_path, output_path, filter_type)
        return jsonify({'scanned': '/static/outputs/' + output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rotate', methods=['POST'])
def rotate():
    data = request.get_json()
    output_filename = data.get('filename')
    direction = data.get('direction', 'right')
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    try:
        rotate_image(output_path, output_path, direction)
        return jsonify({'scanned': '/static/outputs/' + output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
