import webbrowser as wb


def get_gdown(raw_http):
    id = raw_http.split('/d/')[1].split('/')[0]
    link = 'https://drive.usercontent.google.com/download?id=' + id
    print(link)
    wb.open(link)


link = 'https://drive.google.com/file/d/1jSX-KgopsQjw5QZ8GRF6q8JXcXV-O2l0/view?usp=drive_link'
get_gdown(link)

print(1 + 12)
