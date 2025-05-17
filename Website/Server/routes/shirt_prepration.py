from flask import Blueprint, request, jsonify
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


from new import prepare_shirt

prepare_shirt_route = Blueprint('prepare_bp', __name__)

@prepare_shirt_route.route('/prepare-shirt', methods=['POST'])
def prepare_shirt_handler():
    from states import path_states
    from states import shirt_state  # import the module, not individual vars for assignment

    data = request.get_json()

    if not data or 'shirt_image_path' not in data:
        return jsonify({"error": "Missing shirt_image_path"}), 400

    shirt_image_path = data['shirt_image_path']
    path_states.PREPARED_IMAGE = os.path.join(path_states.IMAGES_DIR,shirt_image_path)

    try:
        (
            shirt_state.shirt_name,
            shirt_state.shirt_no_bg,
            shirt_state.fps_history,
            shirt_state.shirt_mask
        ) = prepare_shirt(path_states.PREPARED_IMAGE)

        return jsonify({"message": "Shirt prepared successfully"}), 200
    except Exception as e:
        print(f"Error in prepare_shirt: {e}")
        return jsonify({"error": str(e)}), 500
