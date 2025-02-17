import os

try :
    import streamlit.components.v1 as components

    # Create a _RELEASE constant. We'll set this to False while we're developing
    # the component, and True when we're ready to package and distribute it.
    # (This is, of course, optional - there are innumerable ways to manage your
    # release process.)
    _RELEASE = False

    # Declare a Streamlit component. `declare_component` returns a function
    # that is used to create instances of the component. We're naming this
    # function "_component_func", with an underscore prefix, because we don't want
    # to expose it directly to users. Instead, we will create a custom wrapper
    # function, below, that will serve as our component's public API.

    # It's worth noting that this call to `declare_component` is the
    # *only thing* you need to do to create the binding between Streamlit and
    # your component frontend. Everything else we do in this file is simply a
    # best practice.

    if not _RELEASE:
        _component_func = components.declare_component(
            # We give the component a simple, descriptive name ("stspeck"
            # does not fit this bill, so please choose something better for your
            # own component :)
            "stspeck",
            # Pass `url` here to tell Streamlit that the component will be served
            # by the local dev server that you run via `npm run start`.
            # (This is useful while your component is in development.)
            url="http://localhost:3001",
        )
    else:
        # When we're distributing a production version of the component, we'll
        # replace the `url` param with `path`, and point it to to the component's
        # build directory:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(parent_dir, "frontend/build")
        _component_func = components.declare_component("stspeck", path=build_dir)

    # Create a wrapper function for the component. This is an optional
    # best practice - we could simply expose the component function returned by
    # `declare_component` and call it done. The wrapper allows us to customize
    # our component's API: we can pre-process its input args, post-process its
    # output value, and add a docstring for users.


    def speck(data, **kwargs):

        """Create a new instance of "stspeck".

        Parameters
        ----------
        data : dict[str, list]
            json of the molecule, default(True)
        bonds : bool
            Enable visualizations of bonds?, default(True)
        atomScale : float
            Atom radius, size of spheres, default(0.24)
        relativeAtomScale : float
            Relative atom radius, default(0.64)
        bondScale : float
            bonds size, size of the tubes connecting atoms, default(0.5)
        brightness : float
            brightness, default(0.5)
        outline : float
            Outline strength, default(0.0)
        spf : float
            Samples per frame, default(32)
        bondThreshold : float
            Bonding radius, defines the max distance for atoms to be connected,
            default(1.2)
        bondShade : float
            bonds shade, default(0.5)
        atomShade : float
            Atoms shade, default(0.5)
        dofStrength : float
            Depth of field strength, default(0.0)
        dofPosition : float
            Depth of field position, default(0.5)

        """
        # Call through to our private component function. Arguments we pass here
        # will be sent to the frontend, where they'll be available in an "args"
        # dictionary.
        #
        # "default" is a special argument that specifies the initial return
        # value of the component before the user has interacted with it.
        component_value = _component_func(
            data=data, 
            bonds=kwargs.get('bonds', True),
            atomScale=kwargs.get('atomScale', 0.24),
            relativeAtomScale=kwargs.get('relativeAtomScale', 0.64),
            bondScale=kwargs.get('bondScale', 0.5),
            brightness=kwargs.get('brightness', 0.5),
            outline=kwargs.get('outline', 1),
            spf=kwargs.get('spf', 0),
            bondThreshold=kwargs.get('bondThreshold', 1.2),
            bondShade=kwargs.get('bondShade', 0.5),
            atomShade=kwargs.get('atomShade', 0.5),
            dofStrength=kwargs.get('dofStrength', 0.0),
            dofPosition=kwargs.get('dofPosition', 0.5),            
            ao=kwargs.get('ao', 0),
            aoRes=kwargs.get('aoRes', 512),
            width=kwargs.get('width', "100%"),
            height=kwargs.get('height', "200px"),
            key=kwargs.get('key', None)
        )

        # We could modify the value returned from the component if we wanted.
        # There's no need to do this in our simple example - but it's an option.
        return component_value

except ImportError:
    pass
