import requests 
import json
import os
import re
import time
from  tkinter  import *


url1 =  'https://c.y.qq.com/soso/fcgi-bin/client_search_cp?&lossless=0&flag_qc=0&p={page}&n=20&w={name}'
url2 = 'https://u.y.qq.com/cgi-bin/musicu.fcg?callback=getplaysongvkey8832349141233764&g_tk=5381&jsonpCallback=getplaysongvkey8832349141233764&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0&data={"req":{"module":"CDN.SrfCdnDispatchServer","method":"GetCdnDispatch","param":{"guid":"1062627656","calltype":0,"userip":""\}\},"req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1062627656","songmid":["0016tIV7p"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}'
url3 = 'http://dl.stream.qqmusic.qq.com/'

def makeSearchUrl1(songmid):
    newurl = 'https://u.y.qq.com/cgi-bin/musicu.fcg?callback=getplaysongvkey8832349141233764&g_tk=5381&jsonpCallback=getplaysongvkey8832349141233764&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0&data={"req":{"module":"CDN.SrfCdnDispatchServer","method":"GetCdnDispatch","param":{"guid":"1062627656","calltype":0,"userip":""}},"req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1062627656","songmid":["' + songmid + '"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}'

    return  newurl
def downLoadMusic(lasturl, name, headers):
    req = requests.get(lasturl, headers= headers, stream= True)
    nename = checkNameValid(name+ ".mp4")
    with open(nename, 'wb') as f:
        for content in req.iter_content(1024):
            f.write(content)

def makeSearchUrl(name,page):
    return url1.format(name=name, page= page)

def center_window(root, width, height):  
    screenwidth = root.winfo_screenwidth()  
    screenheight = root.winfo_screenheight()  
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width)/2, (screenheight - height)/2)    
    root.geometry(size) 

def exitFun():
    exit(1)

def search():
    name = entry.get()
    page = int(entry1.get())
    workdir = './' + name
    if not os.path.exists(workdir) :
        os.mkdir(workdir)
    os.chdir(workdir)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
    }

    for i in range(1,page+1):
        searchUrl =  makeSearchUrl(name, str(i))
        req = requests.get(searchUrl, headers= headers)
        json1 = json.loads(req.content.decode('utf-8')[9:][:-1])
        newdata = json1['data']['song']['list']
        dataDict = {}
        for item in newdata:
            dataDict[item['songname']] = item['songmid']
            newurl =  makeSearchUrl1(item['songmid'])
            req1 = requests.get(newurl, headers= headers)
            json2 = json.loads(req1.content.decode('utf-8')[32:][:-1])
            if len(json2['req_0']['data']['midurlinfo']) <= 0 :
                break
            purl = json2['req_0']['data']['midurlinfo'][0]['purl']
            lasturl = url3 + purl
            downLoadMusic(lasturl, item['songname'], headers)
    os.chdir('../')
    # msgbox.showinfo("下载完成")

def checkNameValid(name=None):
    """
    检测Windows文件名称！
    """
    if name is None:
        print("name is None!")
        return
    reg = re.compile(r'[\\/:*?"<>|\r\n]+')
    valid_name = reg.findall(name)
    if valid_name:
        for nv in valid_name:
            name = name.replace(nv, "_")
    return name




root=Tk()
label  =    Label(root, text= '明星名字：')
label2  =    Label(root, text= '页数：')
label.grid(column=0, row=0)
label2.grid(column=0, row=1)
button = Button(root, text= "搜索", command=search)
button1 = Button(root, text= "退出", command=exitFun)
entry = Entry(root)
entry.grid(column=1, row=0)
entry1 = Entry(root)
entry1.grid(column=1, row=1)
button.grid(column=0, row=2)
button1.grid(column=1, row=2)

# canvas=Canvas(root)
# canvas.create_text(100,100)     #填充颜色#这里面，100，100就是text的位置
# im=PhotoImage(file=r"1.png")     #载入图片这里只有png和gif格式的可以，而jpg格式的却不可以。
# canvas.create_image(150,150,image=im)       #载入图片
# canvas.grid(column=1, row=3)
root.title("search music")
center_window(root, 300, 300)
# root.iconbitmap('x.ico')
root.mainloop()


