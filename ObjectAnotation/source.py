import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    big_image = cv2.imread("./2.png")
    big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB)

    #98:817 543:1982
    cropped_image = big_image[260:980, 125:1565]

    net = cv2.dnn.readNetFromCaffe("../GoogleNet/bvlc_googlenet.prototxt", "../GoogleNet/bvlc_googlenet.caffemodel")

    with open("../GoogleNet/classification_classes_ILSVRC2012.txt", 'r') as f:
        class_labels = f.read().strip().split("\n")

    sizes = [((180, 180), 180), ((360, 360), 360), ((720, 720), 720)] #velicine male, srednje i velike slike macaka
    confidence_threshold = 60
    margin = 3

    for window_size, step_size in sizes:
        for y in range(0, cropped_image.shape[0] - window_size[1] + 1, step_size): #po y gleda u slici da li ima velicina prozora 180, 360...
            for x in range(0, cropped_image.shape[1] - window_size[0] + 1, step_size): #isto za x
                roi = cropped_image[y:y + window_size[1], x:x + window_size[0]] #doda se na koordinatu mali, veliki ili srednji prozor
                blob = cv2.dnn.blobFromImage(roi, 1, (224, 224), (104, 117, 123)) #nalazi blok fiksne velicine koju prima neuronska
                net.setInput(blob)
                preds = net.forward() #salje se mrezi
                class_idx = np.argmax(preds[0])
                confidence = preds[0][class_idx] * 100

                if confidence >= confidence_threshold: #verovatnoca prelazi treshold
                    class_label = class_labels[class_idx] #u spisku klasa gleda da li je taj trenutni prepoznat dog
                    if 'dog' in class_label:
                        cv2.rectangle(cropped_image, (x + margin, y + margin), (x + window_size[0] - margin, y + window_size[1] - margin), (255, 255, 0), 2)
                        cv2.putText(cropped_image, "DOG", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    elif 'cat' in class_label:
                        cv2.rectangle(cropped_image, (x + margin, y + margin), (x + window_size[0] - margin, y + window_size[1] - margin), (255, 0, 0), 2)
                        cv2.putText(cropped_image, "CAT", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # cv2.imwrite('output.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    plt.imshow(cropped_image)
    plt.axis("off")
    plt.show()
