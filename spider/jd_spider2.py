import pandas as pd
import urllib.request as req
import json
import sys
import time
import random
from tkinter import *
import tkinter.messagebox
import requests
from bs4 import BeautifulSoup
import re
import chardet
import os 

def MyThread(url,itemid, page,callback):
    html1 =  getHtml(url)
    imageurls2 = BeautifulSoup(html1, 'html.parser')
    namestr = imageurls2.find('ul', {'class' :'parameter2 p-parameter-list'}).find('li',string= re.compile("商品名称：.*")).string
    namestr = namestr.replace('商品名称', '')
    namestr = namestr.replace(':', '')
    namestr = namestr.replace('：', '')
    info = {
        'name':namestr.strip(),
        'itemid': itemid
    
    }
    
    productId = int(info['itemid'])
    print(url)
    JDC = JDCommentsCrawler(info['name'],productId,callback,page)
    JDC.concatLinkParam()
    JDC.crawler()

def getHtml(url):
#携带请求头信息，否则无法获取数据
    headers={"User-Agent":"Mozilla/5.0"}
    response=requests.get(url,headers=headers)
    encoding =  chardet.detect(response.content).get('encoding')
    return response.content.decode(encoding,'ignore')


    return response.content.decode('utf-8')

def getItemList():
    u1 = "https://search.jd.com/Search?keyword=%E5%BC%BA%E7%94%9F%E8%87%AA%E8%90%A5%E6%97%97%E8%88%B0%E5%BA%97&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&bs=1&suggest=3.def.0.V03&wq=%E5%BC%BA%E7%94%9F%E6%97%97%E8%88%B0&ev=exbrand_%E5%BC%BA%E7%94%9F%EF%BC%88Johnson%EF%BC%89%5E&stock=1&page="
    u2 = "&s=61&click=0"
    urls = []
    for i in range(5):
        urls.append(u1 + str(2*i+1)+u2)
    imageurls1=[]
    for url in urls:
        html =  getHtml(url)
        imageurls = BeautifulSoup(html, 'html.parser')
        for one in imageurls.find_all(href=re.compile('.*#comment')):
            content = re.split('[./]', one.attrs['href'])[-2]
            imageurls1.append(content)
    return imageurls1

 
class JDCommentsCrawler:
    
    def __init__(self,name, productId=None,callback=None,page=1,score=0,sortType=5,pageSize=10):
        self.productId = productId #商品ID
        self.score = score # 评论类型（好：3、中：2、差：1、所有：0）
        self.sortType = sortType # 排序类型（推荐：5、时间：6）
        self.pageSize = pageSize # 每页显示多少条记录（默认10）
        self.callback = callback # 回调函数，每个商品都不一样
        self.page = page
        self.name = name
        self.locationLink = 'https://sclub.jd.com/comment/productPageComments.action'
        self.paramValue = {
            'callback':self.callback,
            'productId':self.productId,
            'score':self.score,
            'sortType':self.sortType,
            'pageSize':self.pageSize,
        }        
        self.locationUrl = None
    def paramDict2Str(self,params):        
        str1 = ''
        for p,v in params.items():
            str1 = str1+ p+'='+str(v)+'&'
        return str1
    def concatLinkParam(self):
        self.locationUrl = self.locationLink+'?'+self.paramDict2Str(self.paramValue)+'isShadowSku=0&fold=1&page=0'
    
    def requestMethod(self):
        headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',            
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Referer':'https://item.jd.com/%d.html'%(self.productId),
            'Host':'sclub.jd.com'          
        }
        reqs = req.Request(self.locationUrl,headers=headers)
        return reqs       
    def showList(self):
        request_m = self.requestMethod()       
        conn = req.urlopen(request_m)
        try:
             return_str = conn.read().decode('gbk')
        except UnicodeDecodeError:
            print(1)
            return_str = conn.read().decode('utf-8')
       
        return_str = return_str[len(self.callback)+1:-2]
        return json.loads(return_str)   
    def requestMethodPage(self,p):
        # 伪装浏览器 ，打开网站
        headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',            
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Referer':'https://item.jd.com/%d.html'%(self.productId),
            'Host':'sclub.jd.com'          
        }
        url = self.locationUrl[:-1]+str(p)
        reqs = req.Request(url,headers=headers)
        return reqs
    def showListPage(self,p):
        request_m = self.requestMethodPage(p)      
        conn = req.urlopen(request_m)
        try:
            return_str = conn.read().decode('gbk')
        except UnicodeDecodeError:
            content = conn.read()
            encoding =  chardet.detect(content).get('encoding')
            return_str = content.decode(encoding,'ignore')
      
        return_str = return_str[len(self.callback)+1:-2]
        return json.loads(return_str)
    def save_csv(self,df):
        # 保存文件
        df.to_excel('%s_%d.xlsx'%(self.name.replace('*', ''), self.productId),encoding='gbk', index=False)
 
    def crawler(self):
        # 把抓取的数据存入CSV文件，设置时间间隔，以免被屏蔽
        dfs = []
        for p in range(self.page):
            json_info = self.showListPage(p)
            tmp_list = []
            
            productCommentSummary = json_info['productCommentSummary']
            productId = productCommentSummary['productId']
            comments = json_info['comments']
            # print(len(comments))
            for com in comments:
                tmp_list.append([com['id'],productId,com['guid'],com['content'],com['creationTime'],com['referenceId'],com['referenceTime'],com['score'],\
                                 com['nickname'],com['userLevelName'],com['isMobile'],com['userClientShow']])
            df = pd.DataFrame(tmp_list,columns=['comment_id','product_id','guid','content','create_time','reference_id','reference_time','score',\
                                               'nickname','user_level','is_mobile','user_client'])
            if p % 5 == 0:
                time.sleep(5) 
            dfs.append(df)
        print(self.productId)
        final_df = pd.concat(dfs,ignore_index=True)
        self.save_csv(final_df)
def call(url,page):
    print(page)
    callback = 'fetchJSON_comment98vv2125' #回调函数
    headers = {
        'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
    }
    reqs = requests.get(url,headers=headers)
    pattern  = re.compile(r'//item.jd.com/\d+.html')
    content = reqs.content.decode('utf-8')
    newresult = list(set(pattern.findall(content)))
    for u in newresult:
        item = re.split('[./]', u)[-2]
        MyThread('https:'+ u,item,page,callback)


def jdComment():
    top = Tk()
    
    label  =  Label(top, text= '页数')
    entry = Entry(top)
    label.place(x=2, y=10)
    entry.place(x=40, y=10)
    def helloCallBack(): 
        workPath = './李施德林/'
        if not os.path.exists(workPath):
            os.mkdir(workPath)
        os.chdir(workPath)
        page = 40
        if entry.get() != "":
            page = int(entry.get())
        call( 'https://mall.jd.com/index-1000076326.html',page)
        tkinter.messagebox.showinfo('提示','爬取数据完毕！')
    def helloCallBack1():
        workPath = './京东旗舰店/'
        if not os.path.exists(workPath):
            os.mkdir(workPath)
        os.chdir(workPath)
        page = 40
        if entry.get() != "":
            page = int(entry.get())   
        call( 'https://mall.jd.com/index-1000002806.html',page)
        tkinter.messagebox.showinfo('提示','爬取数据完毕！')
    def exitFun():
        exit(0)

    B = Button(top, text ="李施德林", command = helloCallBack)

    B.place(x=240, y=2)

    B1 = Button(top, text ="强生旗舰", command = helloCallBack1)

    B1.place(x=320, y=2)

    label0  =    Label(top, text= '商店网址')
    entry0 = Entry(top)
    # entry0.insert(0, '填京东店铺网址')
    label0.place(x=2, y=58)
    entry0.place(x=30, y=58)

    label3  =    Label(top,text= '页数')
    label3.place(x=180, y=58)
    entry3 = Entry(top)
    entry3.place(x=220, y=58)

    def anyCallBack():
        workPath = './京东评论/'
        if not os.path.exists(workPath):
            os.mkdir(workPath)
        os.chdir(workPath)
        page = 40
        if entry3.get() != "":
            page = int(entry3.get())  
        url = entry0.get()
        call( url,page)
        tkinter.messagebox.showinfo('提示','爬取数据完毕！')

    B3 = Button(top, text ="开始", command = anyCallBack)

    B3.place(x=370, y=50)
    B2 = Button(top, text ="退出", command = exitFun)

    B2.place(x=200, y=84)
    sw = top.winfo_screenwidth()
    sh = top.winfo_screenheight()
    ww = 440
    wh = 130
    x = (sw-ww) / 2

    y = (sh-wh) / 2

    top.geometry("%dx%d+%d+%d" %(ww,wh,x,y))
    # top.iconbitmap('x.ico')
    top.title("京东评论爬虫")
    top.mainloop()

 
jdComment()
