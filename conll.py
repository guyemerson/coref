import os, logging, re, codecs

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(''message)s')

__author__="kevin heffernan"

class ConllDocument:
    def __init__(self,content):
        self.sents = []
        self.pos_tags = []
        self.speakers = []
        self.coref_chain = {}
        self.open_mentions = []
        self.content = content

    def add_sent(self,sent):
        self.sents.append(sent)

    def add_coref_chain(self,chain_id):
        self.coref_chain[chain_id]=[]

    def add_mention(self,mention):
        chain_id = mention.chain_id
        self.coref_chain[chain_id].append(mention)
         
    def update_coref_chains(self,sent_index,token_index,token,chains):
        for chain in chains.split("|"):
            chain_id = re.sub("\(|\)","",chain)
            if chain_id not in self.coref_chain:
                self.add_coref_chain(chain_id)
            if chain.startswith("(") and chain.endswith(")"):
                mention = Mention(chain_id,sent_index,token_index,token_index)
                mention.update(token_index,token)
                self.add_mention(mention)
            elif chain.startswith("("):
                mention = Mention(chain_id,sent_index,token_index,token_index)
                self.open_mentions.append(mention)
            elif chain.endswith(")"):
                mention = [m for m in self.open_mentions if m.chain_id == chain_id][0]
                mention.update(token_index,token)
                self.add_mention(mention)
                self.open_mentions.remove(mention)
        for mention in self.open_mentions:
            mention.update(token_index,token)

    def get_document_tokens(self):
        """
        returns document as list of tokens
        """
        return [token for sent in self.sents for token in sent]


class Mention:
    def __init__(self,chain_id,sent_index,start_index,end_index):
        self.chain_id = chain_id
        self.sent_index = sent_index
        self.start_index = start_index
        self.end_index = end_index
        self.tokens = []

    def get_indices(self):
        return (self.sent_index,self.start_index,self.end_index)

    def get_document_index(self,conll_doc):
        """ 
        returns index for mention within entire document
        """
        start_token_index = sum([len(conll_doc.sents[i]) for i in range(0,self.sent_index)])
        return (start_token_index+self.start_index,start_token_index+self.end_index)

    def update(self,end_index,token):
        self.end_index = end_index
        self.tokens.append(token)

class ConllCorpusReader:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.train_conll_docs = []
        self.dev_conll_docs=[]
        self.test_conll_docs = []

    def add_conll_doc(self,conll_doc,option):
        if option == "train":
            self.train_conll_docs.append(conll_doc)
        elif option == "development":
            self.dev_conll_docs.append(conll_doc)
        elif option == "test":
            self.test_conll_docs.append(conll_doc)

    def get_conll_docs(self,option):
        if option == "train":
            return self.train_conll_docs
        elif option == "development":
            return self.dev_conll_docs
        elif option == "test":
            return self.test_conll_docs
        
    def get_doc_list(self,option):
        return sorted([os.path.join(r,d) for r, _, ds in os.walk(self.root_dir+"v4/data/"+option+"/data/english/annotations") for d in ds if d.endswith("gold_conll")])

    def get_conll_doc_parts(self,doc):
        conll_doc_parts = []
        for line in codecs.open(doc,"r","utf-8"):
            if line.startswith("#begin document"):
                content = []
            elif line.startswith("#end document"):
                conll_doc_parts.append(ConllDocument(content))
            else:
                content.append(line)
        return conll_doc_parts

    def parse_conll_doc(self,conll_doc):
        sent = []
        sent_index = 0
        for line in conll_doc.content:
            if line=="\n":
                conll_doc.add_sent(sent)
                sent = []
                sent_index+=1
            else:
                token_index,token,pos,speaker,chains = int(line.split()[2]),line.split()[3],line.split()[4],line.split()[9],line.split()[-1]
                conll_doc.update_coref_chains(sent_index,token_index,token,chains)
                sent.append(token) if token != "/." else sent.append(".")
                conll_doc.pos_tags.append(pos)
                conll_doc.speakers.append(speaker)

    def parse_docs(self,option):
        logging.info("parsing "+option+" docs")
        for doc in self.get_doc_list(option):
            for conll_doc in self.get_conll_doc_parts(doc):
                self.parse_conll_doc(conll_doc)
                self.add_conll_doc(conll_doc,option)
            
    def parse_corpus(self):
        self.parse_docs("train")
        self.parse_docs("development")
        self.parse_docs("test")
        logging.info("parsing complete")
        return self
