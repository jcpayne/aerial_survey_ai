import os
from jinja2 import Environment, PackageLoader


class Writer:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('pascal_voc_writer', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }
        
    #Full filepath
    def changePath(self,newpath):
        self.template_parameters['path'] = newpath
        
    #Just the filename, including extension
    def changeFilename(self,newfilename):
        self.template_parameters['filename'] = newfilename
        
    #The parent folder of the file
    def changeFolder(self,newfolder):
        self.template_parameters['folder'] = newfolder    

    #attributes= a list of dicts; each dict is one attribute and looks like: {'name':<name>, 'value':<value>}    
    def addObject(self, name, xmin, ymin, xmax, ymax, score=-1, pose='Unspecified', truncated=0, difficult=0,attributes=[]):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'score':score,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
            'attributes': attributes
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)
