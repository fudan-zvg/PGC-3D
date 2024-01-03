'use strict';

(function () {
    // Create mesh holders.
    let sharedAttributes = {
        ar: true,
        "ar-modes": "webxr scene-viewer quick-look",
        loading: "lazy",
        reveal: "manual",
        // crossorigin: "anonymous",
        style: "height: 300px; width: 100%;",
        "camera-controls": true,
        "touch-action": "pan-y",
        "shadow-intensity": "1",
        exposure: "1"
    };

    let meshAttributes = {
        m1: {
            src: "/assets/glb/JKfigure_m1.glb",
            poster: "/assets/poster/JKfigure_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "JKfigure m1",
            shortCaption: "JKfigure m1"
        },
        m2: {
            src: "/assets/glb/JKfigure_m2.glb",
            poster: "/assets/poster/JKfigure_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "JKfigure m2",
            shortCaption: "JKfigure m2"
        },
        m3: {
            src: "/assets/glb/JKfigure_m3.glb",
            poster: "/assets/poster/JKfigure_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "JKfigure m3",
            shortCaption: "JKfigure m3"
        },
        m0: {
            src: "/assets/glb/angry_cat1_m0.glb",
            poster: "/assets/poster/angry_cat1_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat1 m0",
            shortCaption: "angry cat1 m0"
        },
        cat1_m1: {
            src: "/assets/glb/angry_cat1_m1.glb",
            poster: "/assets/poster/angry_cat1_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat1 m1",
            shortCaption: "angry cat1 m1"
        },
        cat1_m2: {
            src: "/assets/glb/angry_cat1_m2.glb",
            poster: "/assets/poster/angry_cat1_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat1 m2",
            shortCaption: "angry cat1 m2"
        },
        cat1_m3: {
            src: "/assets/glb/angry_cat1_m3.glb",
            poster: "/assets/poster/angry_cat1_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat1 m3",
            shortCaption: "angry cat1 m3"
        },
        cat2_m1: {
            src: "/assets/glb/angry_cat2_m1.glb",
            poster: "/assets/poster/angry_cat2_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat2 m1",
            shortCaption: "angry cat2 m1"
        },
        cat2_m2: {
            src: "/assets/glb/angry_cat2_m2.glb",
            poster: "/assets/poster/angry_cat2_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat2 m2",
            shortCaption: "angry cat2 m2"
        },
        cat2_m3: {
            src: "/assets/glb/angry_cat2_m3.glb",
            poster: "/assets/poster/angry_cat2_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "angry cat2 m3",
            shortCaption: "angry cat2 m3"
        },
        man_m0: {
            src: "/assets/glb/armor_man_m0.glb",
            poster: "/assets/poster/armor_man_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "armor man m0",
            shortCaption: "armor man m0"
        },
        man_m1: {
            src: "/assets/glb/armor_man_m1.glb",
            poster: "/assets/poster/armor_man_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "armor man m1",
            shortCaption: "armor man m1"
        },
        man_m2: {
            src: "/assets/glb/armor_man_m2.glb",
            poster: "/assets/poster/armor_man_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "armor man m2",
            shortCaption: "armor man m2"
        },
        man_m3: {
            src: "/assets/glb/armor_man_m3.glb",
            poster: "/assets/poster/armor_man_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "armor man m3",
            shortCaption: "armor man m3"
        },
        car_m0: {
            src: "/assets/glb/castle_on_car_m0.glb",
            poster: "/assets/poster/castle_on_car_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "castle on car m0",
            shortCaption: "castle on car m0"
        },
        car_m1: {
            src: "/assets/glb/castle_on_car_m1.glb",
            poster: "/assets/poster/castle_on_car_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "castle on car m1",
            shortCaption: "castle on car m1"
        },
        car_m2: {
            src: "/assets/glb/castle_on_car_m2.glb",
            poster: "/assets/poster/castle_on_car_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "castle on car m2",
            shortCaption: "castle on car m2"
        },
        car_m3: {
            src: "/assets/glb/castle_on_car_m3.glb",
            poster: "/assets/poster/castle_on_car_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "castle on car m3",
            shortCaption: "castle on car m3"
        },
        sword_m0: {
            src: "/assets/glb/dragon_sword_m0.glb",
            poster: "/assets/poster/dragon_sword_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "dragon sword m0",
            shortCaption: "dragon sword m0"
        },
        sword_m1: {
            src: "/assets/glb/dragon_sword_m1.glb",
            poster: "/assets/poster/dragon_sword_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "dragon sword m1",
            shortCaption: "dragon sword m1"
        },
        sword_m2: {
            src: "/assets/glb/dragon_sword_m2.glb",
            poster: "/assets/poster/dragon_sword_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "dragon sword m2",
            shortCaption: "dragon sword m2"
        },
        sword_m3: {
            src: "/assets/glb/dragon_sword_m3.glb",
            poster: "/assets/poster/dragon_sword_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "dragon sword m3",
            shortCaption: "dragon sword m3"
        },
        helmet_m0: {
            src: "/assets/glb/luffy_helmet_m0.glb",
            poster: "/assets/poster/luffy_helmet_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "luffy helmet m0",
            shortCaption: "luffy helmet m0"
        },
        helmet_m1: {
            src: "/assets/glb/luffy_helmet_m1.glb",
            poster: "/assets/poster/luffy_helmet_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "luffy helmet m1",
            shortCaption: "luffy helmet m1"
        },
        helmet_m2: {
            src: "/assets/glb/luffy_helmet_m2.glb",
            poster: "/assets/poster/luffy_helmet_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "luffy helmet m2",
            shortCaption: "luffy helmet m2"
        },
        helmet_m3: {
            src: "/assets/glb/luffy_helmet_m3.glb",
            poster: "/assets/poster/luffy_helmet_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "luffy helmet m3",
            shortCaption: "luffy helmet m3"
        },
        camera_m0: {
            src: "/assets/glb/man_camera_m0.glb",
            poster: "/assets/poster/man_camera_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "man camera m0",
            shortCaption: "man camera m0"
        },
        camera_m1: {
            src: "/assets/glb/man_camera_m1.glb",
            poster: "/assets/poster/man_camera_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "man camera m1",
            shortCaption: "man camera m1"
        },
        camera_m2: {
            src: "/assets/glb/man_camera_m2.glb",
            poster: "/assets/poster/man_camera_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "man camera m2",
            shortCaption: "man camera m2"
        },
        old_man_m0: {
            src: "/assets/glb/old_man_m0.glb",
            poster: "/assets/poster/old_man_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "old man m0",
            shortCaption: "old man m0"
        },
        old_man_m1: {
            src: "/assets/glb/old_man_m1.glb",
            poster: "/assets/poster/old_man_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "old man m1",
            shortCaption: "old man m1"
        },
        old_man_m2: {
            src: "/assets/glb/old_man_m2.glb",
            poster: "/assets/poster/old_man_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "old man m2",
            shortCaption: "old man m2"
        },
        old_man_m3: {
            src: "/assets/glb/old_man_m3.glb",
            poster: "/assets/poster/old_man_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "old man m3",
            shortCaption: "old man m3"
        },
        shield_m0: {
            src: "/assets/glb/panda_spear_shield_m0.glb",
            poster: "/assets/poster/panda_spear_shield_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "panda spear shield m0",
            shortCaption: "panda spear shield m0"
        },
        shield_m1: {
            src: "/assets/glb/panda_spear_shield_m1.glb",
            poster: "/assets/poster/panda_spear_shield_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "panda spear shield m1",
            shortCaption: "panda spear shield m1"
        },
        shield_m2: {
            src: "/assets/glb/panda_spear_shield_m2.glb",
            poster: "/assets/poster/panda_spear_shield_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "panda spear shield m2",
            shortCaption: "panda spear shield m2"
        },
        shield_m3: {
            src: "/assets/glb/panda_spear_shield_m3.glb",
            poster: "/assets/poster/panda_spear_shield_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "panda spear shield m3",
            shortCaption: "panda spear shield m3"
        },
        archer_m0: {
            src: "/assets/glb/wolfman_archer_m0.glb",
            poster: "/assets/poster/wolfman_archer_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wolfman archer m0",
            shortCaption: "wolfman archer m0"
        },
        archer_m1: {
            src: "/assets/glb/wolfman_archer_m1.glb",
            poster: "/assets/poster/wolfman_archer_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wolfman archer m1",
            shortCaption: "wolfman archer m1"
        },
        archer_m2: {
            src: "/assets/glb/wolfman_archer_m2.glb",
            poster: "/assets/poster/wolfman_archer_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wolfman archer m2",
            shortCaption: "wolfman archer m2"
        },
        archer_m3: {
            src: "/assets/glb/wolfman_archer_m3.glb",
            poster: "/assets/poster/wolfman_archer_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wolfman archer m3",
            shortCaption: "wolfman archer m3"
        },
        wooden_car_m0: {
            src: "/assets/glb/wooden_car_m0.glb",
            poster: "/assets/poster/wooden_car_m0.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wooden car m0",
            shortCaption: "wooden car m0"
        },
        wooden_car_m1: {
            src: "/assets/glb/wooden_car_m1.glb",
            poster: "/assets/poster/wooden_car_m1.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wooden car m1",
            shortCaption: "wooden car m1"
        },
        wooden_car_m2: {
            src: "/assets/glb/wooden_car_m2.glb",
            poster: "/assets/poster/wooden_car_m2.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wooden car m2",
            shortCaption: "wooden car m2"
        },
        wooden_car_m3: {
            src: "/assets/glb/wooden_car_m3.glb",
            poster: "/assets/poster/wooden_car_m3.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
            caption: "wooden car m3",
            shortCaption: "wooden car m3"
        },


    };

    let meshRows = [
        ['m1', 'm2', 'm3'],
        // ['m0', 'cat1_m1', 'cat1_m2', 'cat1_m3'],
        // ['cat2_m1', 'cat2_m2', 'cat2_m3'],
        // ['man_m0', 'man_m1', 'man_m2', 'man_m3'],
        // ['car_m0', 'car_m1', 'car_m2', 'car_m3'],
        // ['sword_m0', 'sword_m1', 'sword_m2', 'sword_m3'],
        // ['helmet_m0', 'helmet_m1', 'helmet_m2', 'helmet_m3'],
        // ['camera_m0', 'camera_m1', 'camera_m2', 'old_man_m0'],
        // ['old_man_m1', 'old_man_m2', 'old_man_m3', 'shield_m0'],
        // ['shield_m1', 'shield_m2', 'shield_m3', 'archer_m0'],
        // ['archer_m1', 'archer_m2', 'archer_m3', 'wooden_car_m0'],
    ];

    let container = document.getElementById("meshContainer");
    meshRows.forEach((meshIds) => {
        let row = document.createElement("DIV");
        row.classList = "row";

        meshIds.forEach((meshId) => {
            let col = document.createElement("DIV");
            col.classList = "col-md-6 col-sm-6 my-auto";

            // Model viewer.
            let model = document.createElement("model-viewer");
            for (const attr in sharedAttributes) {
                if (attr != "caption" && attr != "shortCaption")
                    model.setAttribute(attr, sharedAttributes[attr]);
            }
            for (const attrCustom in meshAttributes[meshId]) {
                if (attrCustom != "caption" && attrCustom != "shortCaption")
                    model.setAttribute(attrCustom, meshAttributes[meshId][attrCustom]);
            }
            model.id = 'mesh-' + meshId;

            // Controls.
            let controls = document.createElement("div");
            controls.className = "controls";
            let buttonLoad = document.createElement("button");
            buttonLoad.classList = "btn btn-primary loads-model";
            buttonLoad.setAttribute("data-controls", model.id);
            buttonLoad.appendChild(document.createTextNode("Load 3D model"));
            // let buttonToggle = document.createElement("button");
            // buttonToggle.classList = "btn btn-primary toggles-texture";
            // buttonToggle.setAttribute("data-controls", model.id);
            // buttonToggle.appendChild(document.createTextNode("Toggle texture"));
            controls.appendChild(buttonLoad);
            controls.appendChild(buttonToggle);

            // Caption.
            let caption = document.createElement("p");
            caption.classList = "caption";
            caption.title = meshAttributes[meshId]["caption"] || "";
            caption.appendChild(document.createTextNode(meshAttributes[meshId]["shortCaption"] || caption.title));

            col.appendChild(model);
            col.appendChild(controls);
            col.appendChild(caption);
            row.appendChild(col);
        });

        container.appendChild(row);
    });

    // Toggle texture handlers.
    document.querySelectorAll('button.toggles-texture').forEach((button) => {
        button.addEventListener('click', () => {
            let model = document.getElementById(button.getAttribute("data-controls"));

            console.log(model.model);
            let material = model.model.materials[0];
            let metallic = material.pbrMetallicRoughness;
            let originalTexture = metallic.pbrMetallicRoughness.baseColorTexture;
            let originalBaseColor = metallic.pbrMetallicRoughness.baseColorFactor;
            console.log('model load', model.model, material, 'metallic', metallic, originalTexture);

            let textureButton = model.querySelector('.toggles-parent-texture');
            console.log('texture button', textureButton);
            // if (originalTexture && textureButton) {
            let textureOn = true;
            textureButton.onclick = () => {
                if (textureOn) {
                    // model.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
                    // model.model.materials[0].pbrMetallicRoughness.setBaseColorFactor([1., 1., 1., 1.]);
                    // textureOn = false;
                    // console.log('toggle texture off');
                } else {
                    // model.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(originalTexture) ;
                    // model.model.materials[0].pbrMetallicRoughness.setBaseColorFactor(originalBaseColor);
                    // textureOn = true;
                    // console.log('toggle texture on');
                }
            };
        });
    });

    // Click to load handlers for 3D meshes.
    document.querySelectorAll('button.loads-model').forEach((button) => {
        button.setAttribute('data-action', 'load');
        button.addEventListener('click', () => {
            // button.classList = button.classList + " disappearing";
            // let model = button.parentElement.parentElement;
            let model = document.getElementById(button.getAttribute("data-controls"));

            if (button.getAttribute('data-action') == 'load') {
                model.dismissPoster();
                button.classList = "btn btn-disabled";
                button.innerHTML = "Hide 3D model";
                button.setAttribute('data-action', 'unload');
            } else {
                model.showPoster();
                button.classList = "btn btn-primary";
                button.innerHTML = "Load 3D model";
                button.setAttribute('data-action', 'load');
            }
            ;
        });
    });
    // document.querySelectorAll('button.toggles-parent-texture').forEach((button) => {
    //     let model = button.parentElement.parentElement;
    //     let originalTexture = model.materials[0].pbrMetallicRoughness.baseColorTexture;
    //     let textureOn = true;
    //     button.addEventListener('click', () => {
    //         if (textureOn) {
    //             model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    //             textureOn = false;
    //         } else {
    //             model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(originalTexture) ;
    //             textureOn = true;
    //         }
    //     });
    // });
})();
