# -*- coding: utf-8 -*-
#import scrapy


# class LianjiaSpider(scrapy.Spider):
#     name = 'lianjia'
#     allowed_domains = [''https://sy.lianjia.com/ershoufang/'']
#     start_urls = ['http://'https://sy.lianjia.com/ershoufang/'/']
#
#     def parse(self, response):
#         pass
from scrapy import Request
from scrapy.spiders import Spider
from lianjiahouse.items import LianjiahouseItem

class house_chart(Spider):
    name = 'lianjia'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36',
    }
    def start_requests(self):
        start_url = 'https://sy.lianjia.com/ershoufang/'
        yield Request(url=start_url,headers=self.headers)

    def parse(self, response):
        house_list = response.xpath('//div[@class="info clear"]/div[@class="title"]/a/@href')

        for node in house_list:
            href = node.extract()
            yield Request(url=href,headers=self.headers,callback=self.parse1)

        #实现翻页
        for i in range(11):
            if i != 0 and i != 1:
                next_url = 'https://sy.lianjia.com/ershoufang/pg%d' % i
                if next_url:
                    yield Request(url=next_url,headers=self.headers,callback=self.parse)

    def parse1(self,response):
        item = LianjiahouseItem()
        item['totalprice'] = response.xpath('//div[@class="price "]/span[@class="total"]/text()').extract()[0]
        item['price'] = response.xpath('//div[@class="unitPrice"]/span[@class="unitPriceValue"]/text()').extract()[0]
        item['house_model'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[1]/text()').extract()[0]
        item['floor'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[2]/text()').extract()[0]
        item['area'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[3]/text()').extract()[0]
        item['structure'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[4]/text()').extract()[0]
        item['space_in'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[5]/text()').extract()[0]
        item['build_type'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[6]/text()').extract()[0]
        item['build_head'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[7]/text()').extract()[0]
        item['build_struc'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[8]/text()').extract()[0]
        item['decorate'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[9]/text()').extract()[0]
        item['proportion'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[10]/text()').extract()[0]
        item['heating_meth'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[11]/text()').extract()[0]
        item['elevator'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[12]/text()').extract()[0]
        item['year_pro'] = response.xpath('//div[@class="base"]/div[@class="content"]/ul/li[13]/text()').extract()[0]

        item['time'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[1]/span[2]/text()').extract()[0]
        item['trans'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[2]/span[2]/text()').extract()[0]
        item['last_trans'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[3]/span[2]/text()').extract()[0]
        item['usage'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[4]/span[2]/text()').extract()[0]
        item['build_pro'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[5]/span[2]/text()').extract()[0]
        item['belonging'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[6]/span[2]/text()').extract()[0]
        item['mortgaga_info'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[7]/span[2]/text()').extract()[0]
        item['room_parts'] = response.xpath('//div[@class="transaction"]/div[@class="content"]/ul/li[8]/span[2]/text()').extract()[0]
        yield item