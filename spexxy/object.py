import logging
import typing


class spexxyObject(object):
    """Base class for all objects in spexxy."""

    def __init__(self, objects: dict = None, log: logging.Logger = None, *args, **kwargs):
        """Initializes a new object.

        Args:
            objects: Dictionary containing other objects.
            log: Logging instance to use.
        """
        self.objects = {} if objects is None else objects
        self._log = log

    @property
    def log(self) -> logging.Logger:
        """Get logger for this object.

        Returns:
            Logger to use for this object.
        """
        return self._log if self._log is not None else logging.getLogger()

    @staticmethod
    def get_class_from_string(class_name: str) -> object:
        """Take a class name as a string and return the actual class

        Args:
            class_name: Name of class.

        Returns:
            Actual class.
        """

        # split parts of class name, i.e. modules and class
        parts = class_name.split('.')

        # join module name
        module_name = ".".join(parts[:-1])

        # import module
        cls = __import__(module_name)

        # fetch class and return it
        for comp in parts[1:]:
            cls = getattr(cls, comp)
        return cls

    @staticmethod
    def create_object(config: dict, log: logging.Logger = None, *args, **kwargs) -> object:
        """Create a new object from a dict.

        Args:
            config: Dictionary with a "class" element to create object from.
            log: Logger to use for new object.

        Returns:
            New object created from config.

        Raises:
            ValueError: Cannot copy config dictionary or no class name given.
        """

        # copy config
        try:
            cfg = dict(config)
        except ValueError:
            log.error('Cannot copy dict: %s', config)
            raise

        # get class name
        class_name = cfg.pop('class', None)
        if class_name is None:
            raise ValueError('No class name given.')

        # create class
        cls = spexxyObject.get_class_from_string(class_name)

        # create object
        return cls(*args, **kwargs, **cfg, log=log)

    def _get_object_from_group(self, name: str, klass, group: str) -> 'spexxyObject':
        """Returns an object from a group.

        Args:
            name: Name of object to retrieve from group.
            klass: Class of object to retrieve from group.
            group: Group to retrieve object from.

        Returns:
            Object with given name and class from the given group.

        Raises:
            ValueError: If object cannot be found.
        """

        # check existence
        if group in self.objects and name in self.objects[group]:
            # get obj and check type
            obj = self.objects[group][name]
            if isinstance(obj, klass):
                return obj
        else:
            # nothing found
            raise ValueError('Object "%s" of class "%s" not found in group "%s".' % (name, klass.__name__, group))

    def _find_name_in_group(self, group: str) -> str:
        """Finds a new unused name in the given group.

        Args:
            group: Group to create a name for.

        Returns:
            New unique name for group-
        """

        # create group if required
        if group not in self.objects:
            self.objects[group] = {}

        # check group name itself
        if group not in self.objects[group]:
            return group

        # otherwise count up
        i = 1
        while True:
            name = '%s-%d' % (group, i)
            if name not in self.objects[group]:
                return name
            i += 1

    def _create_object(self, name: str, definition: dict, klass, group: str, log: logging.Logger = None) \
            -> 'spexxyObject':
        """Creates a single object from the given definition.

        Args:
            name: Name for new object to create.
            definition: Dictionary containing the description of the object to create, especially a "class" element.
            klass: Class the new object should have.

        Returns:
            New object created from definition.

        Raises:
            ValueError: If newly created class is not of given class.
        """

        # create object
        obj = spexxyObject.create_object(definition, log=log)

        # check type
        if not isinstance(obj, klass):
            raise ValueError('Newly created object "%s" is of type "%s", should be "%s".',
                             name, obj.__class__.__name__, klass.__name__)

        # add to objects dict
        if group not in self.objects:
            self.objects[group] = {}
        self.objects[group][name] = obj

        # return it
        return obj

    def get_objects(self, definition: typing.Union[object, str, list, dict], klass, group: str,
                    log: logging.Logger = None, single: bool = False) -> typing.Union[list, 'spexxyObject', None]:
        """Get either a single object or a list of objects from the given definition

        Args:
            definition: Definition of a single object or a list of objects or an object itself.
            klass: Expected class for object.
            group: Group to get object from or put it into.
            log: Logger for newly created object.
            single: Always return single element or None instead of list.

        Returns:
            Either a single object or a list of objects or None

        Raises:
            ValueError: If object could not be retrieved/created.
        """

        # if no logger is given, use mine
        if log is None:
            log = self._log

        # init list of objs
        objs = []

        # decide on what we got in definition
        if definition is None:
            # just nothing
            pass

        elif isinstance(definition, klass):
            # got the correct type from the beginning!
            objs = [definition]

        elif isinstance(definition, str):
            # definition must be a name of an object in the given group
            objs = [self._get_object_from_group(definition, klass, group)]

        elif isinstance(definition, list):
            # definition can be a list of strings with names of objects in the given group or a list of defs
            # loop all definitions and get them
            for defn in definition:
                if isinstance(defn, str):
                    obj =self._get_object_from_group(defn, klass, group)
                elif isinstance(defn, dict) and 'class' in defn:
                    # single def, find a good name and create it
                    name = self._find_name_in_group(group)
                    obj = self._create_object(name, defn, klass, group, log=log)
                else:
                    raise ValueError('Invalid definition.')

                # add it
                objs.append(obj)

        elif isinstance(definition, dict):
            # is this just a single def or multiple?
            if 'class' in definition:
                # single def, find a good name and create it
                name = self._find_name_in_group(group)
                objs = [self._create_object(name, definition, klass, group, log=log)]

            else:
                # definition must be a dict of name->definition pairs for objects to create
                if any(['class' not in d for d in definition.values()]):
                    raise ValueError('Dict must contain name->definition pairs.')

                # create them
                objs = [self._create_object(name, defn, klass, group, log=log) for name, defn in definition.items()]

        else:
            raise ValueError('Unknown type for object definition.')

        # single or list?
        return (None if len(objs) == 0 else objs[0]) if single else objs


__all__ = ['spexxyObject']
