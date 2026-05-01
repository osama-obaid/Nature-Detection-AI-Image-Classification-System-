import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# تحميل الموديل
model = load_model('cnn_saved_model.h5')

# نفس ترتيب الكلاسات اللي دربت بها الموديل
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'see', 'street']

# ألوان مميزة لكل كلاس
class_colors = {
    'buildings': (0, 0, 255),   # أحمر
    'forest': (0, 255, 0),      # أخضر
    'glacier': (255, 255, 255), # أبيض
    'mountain': (128, 0, 128),  # بنفسجي
    'see': (255, 0, 0),         # أزرق
    'street': (128, 128, 128)   # رمادي
}

def preprocess_frame(frame, target_size=(150,150), convert_bgr2rgb=True):
    if convert_bgr2rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if (frame.shape[0] > target_size[1] or frame.shape[1] > target_size[0]) else cv2.INTER_LINEAR
    img = cv2.resize(frame, target_size, interpolation=interp)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(frame, model, class_labels, top_k=3):
    x = preprocess_frame(frame)
    preds = model.predict(x, verbose=0)[0]
    if (preds >= 0).all() and np.isclose(preds.sum(), 1.0, atol=1e-3):
        probs = preds
    else:
        probs = tf.nn.softmax(preds).numpy()
    idxs = np.argsort(probs)[::-1][:top_k]
    results = [(class_labels[i], float(probs[i])) for i in idxs]
    return results

# محاولة فتح الكاميرا 1 أولاً
def switch_camera(cap, current_cam):
    """دالة لتبديل الكاميرا"""
    cap.release()
    new_cam = 1 if current_cam == 0 else 0
    cap = cv2.VideoCapture(new_cam)
    if not cap.isOpened():
        print(f"⚠️ الكاميرا {new_cam} غير متوفرة، الرجوع للكاميرا {current_cam}")
        cap = cv2.VideoCapture(current_cam)  # رجوع للكاميرا القديمة
        return cap, current_cam
    print(f"✅ تم التبديل إلى الكاميرا {new_cam}")
    return cap, new_cam


# البداية بالكاميرا 0
cam_index = 0
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print("⚠️ الكاميرا 0 غير متوفرة، المحاولة مع الكاميرا 1 ...")
    cap = cv2.VideoCapture(1)
    cam_index = 1 if cap.isOpened() else -1

if cam_index == -1:
    print("❌ لم يتم العثور على أي كاميرا")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = predict_frame(frame, model, class_labels, top_k=3)

    # النتيجة الأولى
    top_label, top_conf = results[0]
    color = class_colors.get(top_label, (0, 255, 0))
    text = f"{top_label} ({top_conf*100:.2f}%)"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)

    # عرض باقي النتائج
    for i, (lbl, conf) in enumerate(results[1:], start=1):
        text = f"{lbl}: {conf*100:.1f}%"
        cv2.putText(frame, text, (10, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("Nature Classification", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('x'), ord('ض')]:
        break
    elif key == ord('c'):  # تبديل الكاميرا
        cap, cam_index = switch_camera(cap, cam_index)

cap.release()
cv2.destroyAllWindows()
