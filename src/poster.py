import urllib.request
import base64
url = 'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'
contents = urllib.request.urlopen(url).read()
data = base64.b64encode(contents)
print(data)