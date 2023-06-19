# configs

README in [English](/README.md) | [简体中文](/readme/README_zh_CN.md)

To make the code more flexible and reusable, we use the following tools to manage the configurations:

- YAML files to record the experiment configurations
- PyYAML to load the YAML files into Python objects
- `EasyDict` to access the items in a (nested) dict as attributes

Through these tools, we can parse experiment configurations from YAML files into Python EasyDict objects. All other interfaces and objects can be instantiated from the EasyDict objects.

## YAML

YAML (YAML Ain't Markup Language) is a human-friendly data serialization language for all programming languages. The advantages of YAML are:

- Consise and intuitive syntax
- Use of spaces / indentation to express structure

This makes YAML more readable and easier to use than other common configuration formats like JSON.

### YAML data

YAML consists of three basic data types:

- **Scalars**: numbers, booleans, strings, null
- **Sequences**: lists, arrays
- **Mappings**: dictionaries, hashes

For example:

```yaml
# In person.yml

# Scalar
name: John Smith
age: 33

# Sequence
fruits:
  - Apple
  - Orange
  - Strawberry

# Mapping
family_members:
  father: Paul Smith
  mother: Janet Smith
  brothers:
    - Jimmy Smith
    - Jack Smith
```

### Configuration file format

Our YAML configuration files follow the following format:

```yaml
# In config.yml

```

## PyYAML

PyYAML is a YAML parser and emitter for Python. The following code snippet shows how to load a YAML file into a Python dict:

```python
>>> import yaml
>>> from pprint import pprint

>>> with open('person.yml') as f:
...     person = yaml.safe_load(f)
>>> pprint(person)
{'age': 33,
 'family_members': {'brothers': ['Jimmy Smith', 'Jack Smith'],
                    'father': 'Paul Smith',
                    'mother': 'Janet Smith'},
 'fruits': ['Apple', 'Orange', 'Strawberry'],
 'name': 'John Smith'}
```

## EasyDict

`EasyDict` supports accessing dictionary values through attributes (recursively), which can make code more concise and easy to read.

```python
>>> from easydict import EasyDict
>>> person = EasyDict(person)

>>> pprint(person.name)
'John Smith'
>>> pprint(person.age)
33
>>> pprint(person.family_members.brothers)
['Jimmy Smith', 'Jack Smith']
```

## References

1. <https://pyyaml.org/wiki/PyYAMLDocumentation>
2. <https://github.com/makinacorpus/easydict>
3. <https://zhuanlan.zhihu.com/p/145173920>
4. <https://zhuanlan.zhihu.com/p/461718976>
