from ultralytics import MTDETR


model = MTDETR("/home/jrkernan/project/RMT-PPAD/ultralytics/best.pt")

model.predict(source='/home/jrkernan/project/RMT-PPAD/test_images/adb4871d-4d063244.jpg', imgsz=(640,640), device=[0], mask_threshold=[0.45,0.9], show_labels=False, save=True, project="runs", name="test1")

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")