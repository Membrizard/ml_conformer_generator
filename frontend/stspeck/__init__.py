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
    def Speck(data, **kwargs):


        """Create a new instance of "stspeck".

        Parameters
        ----------
        data : str
            xyz model, default(True)
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
            aoRes=kwargs.get('aoRes', 256),
            width=kwargs.get('width', "100%"),
            height=kwargs.get('height', "200px"),
            key=kwargs.get('key', None)
        )

        # We could modify the value returned from the component if we wanted.
        # There's no need to do this in our simple example - but it's an option.
        return component_value


    # Add some test code to play with the component while it's in development.
    # During development, we can run this just as we would any other Streamlit
    # app: `$ streamlit run stspeck/__init__.py`
    # if not _RELEASE:
#         import streamlit as st
#
#         num_clicks = Speck(
#             '''40
# Coordinates from ORCA-job ./DSI-PABA-Me-FTIR/DSI-PABA-Me
#   H   0.98937496325200     -1.17478886328261      2.27058125848368
#   C   0.53531596739715     -0.76282427915773      1.36680842143398
#   C   -0.78938552381275     -0.31798843028063      1.36269226290030
#   H   -1.38607185049040     -0.36746422119338      2.27420481912495
#   C   -1.37478709409532      0.18686460687156      0.18829473157390
#   C   -0.63541555443116      0.25562383844070     -1.00251181516760
#   H   -1.10948761989374      0.63348515891456     -1.91078113368286
#   C   0.68643604109322     -0.17456607777946     -0.98255974405120
#   C   1.26287724349208     -0.67818881365359      0.18133270013379
#   S   1.76178347950674     -0.12296396524355     -2.41672993210774
#   S   2.97348584949572     -1.19524105573980      0.03272481541705
#   N   2.94161268615442     -1.18598864302905     -1.70370794751065
#   C   -2.79950483400440      0.65932256413139      0.14035262099225
#   O   -3.32497425054190      1.10119604747252     -0.85277063982672
#   O   -3.41806523603810      0.53296065111082      1.32304521534877
#   C   -4.78422270217983      0.95671446869928      1.37491358415418
#   H   -4.87096782487068      2.01963066378706      1.10405946172599
#   H   -5.11329430024711      0.79549385229867      2.40856168674904
#   H   -5.39872346828285      0.36576784805343      0.67904488430394
#   O   1.14689081415204     -0.80961510795539     -3.54381995239774
#   O   2.30488837165075      1.22216936221585     -2.58514185882761
#   O   3.13105806044837     -2.57232356155918      0.48207885005225
#   O   3.85684388051777     -0.15323396412433      0.54795967114354
#   C   4.19119209640559     -1.36691614221876     -2.38732788934871
#   C   5.12133176079697     -0.31996021114267     -2.47788721130884
#   C   4.46445213661466     -2.62078760898587     -2.95050041330210
#   C   5.67354598307080     -2.83094942172912     -3.61108213399133
#   C   6.60915879528154     -1.78816027083424     -3.71107190539074
#   C   6.32587890727401     -0.53598361010082     -3.14414712630351
#   H   4.88983109518368      0.64604731591232     -2.02874848384634
#   H   3.72182913808606     -3.41519683946282     -2.85986598544254
#   H   5.89997096100231     -3.80192156931734     -4.05269942711463
#   H   7.06844490663233      0.25913073885430     -3.23363744992666
#   C   7.92345209331541     -1.95685772054555     -4.40888733430767
#   O   8.75491482319406     -1.08555232388284     -4.51373730965928
#   O   8.08396283825505     -3.19447588582287     -4.91251692435344
#   C   9.31032385550847     -3.44256073280942     -5.60019500486842
#   H   9.26905038981127     -4.48752738213999     -5.93207166680964
#   H   10.17150374026012     -3.28607315551632     -4.93257007126225
#   H   9.41568938103552     -2.77007725925507     -6.46543562272933''',
#             height = "800px",
#             atomScale = 0.5
#         )
except ImportError:
    pass
