import scrapy
from chaidocs_crawler.items import ChaidocsCrawlerItem

class ChaiDocsSpider(scrapy.Spider):
    name = 'chaidocs'
    allowed_domains = ['docs.chaicode.com']
    start_urls = ['https://docs.chaicode.com/youtube/getting-started/']

    def parse(self, response):
        item = ChaidocsCrawlerItem()

        item['url'] = response.url
        item['title'] = response.css('title::text').get()
        item['content'] = response.xpath('//body').get()
        yield item

        # Follow all internal links recursively
        for href in response.css('a::attr(href)').getall():
            if href.startswith('/'):
                href = response.urljoin(href)
            if "docs.chaicode.com" in href:
                yield response.follow(href, self.parse)

