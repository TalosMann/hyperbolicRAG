import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            # The namespaces are usually standard in Word, we can just strip them or use them
            WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            PARA = WORD_NAMESPACE + 'p'
            TEXT = WORD_NAMESPACE + 't'
            
            paragraphs = []
            for paragraph in tree.iter(PARA):
                texts = [node.text for node in paragraph.iter(TEXT) if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    text = read_docx(r'D:\Projects\hyperbolic\HyperRAG_vs_HierarchicalRAG_Evaluation_Report.docx')
    with open(r'D:\Projects\hyperbolic\report_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
