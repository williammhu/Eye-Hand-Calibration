import zxing

print("zxing module file:", zxing.__file__)

reader = zxing.BarCodeReader()
print("reader OK")

# 换成你磁盘上一张清晰的条码 / QR 图片
img_path = r"C:\Users\24065\Desktop\barcode.jpg"

result = reader.decode(img_path)
print("raw result object:", result)

if result is not None:
    print("parsed:", getattr(result, "parsed", None))
    print("raw:", getattr(result, "raw", None))
