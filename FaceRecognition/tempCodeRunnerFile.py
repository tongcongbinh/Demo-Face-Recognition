cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)
        cv2.putText(frame, name, (x2, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)