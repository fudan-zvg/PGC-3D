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
            src: "/assets/glb/car.glb",
            poster: "/assets/poster/car.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/lightroom_14b.hdr",
            caption: "A wooden car",
            shortCaption: "A wooden car"
        },
        m2: {
            src: "/assets/glb/horse.glb",
            poster: "/assets/poster/horse.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/lightroom_14b.hdr",
            caption: "An astronaut riding a horse",
            shortCaption: "An astronaut riding a horse"
        },
        m3: {
            src: "/assets/glb/panda.glb",
            poster: "/assets/poster/panda.png",
            "environment-image": "https://modelviewer.dev/shared-assets/environments/lightroom_14b.hdr",
            caption: "A panda is dressed in armor, holding a spear in one hand and a shield in the other",
            shortCaption: "A panda holding a spear and a shield"
        },


    };

    let meshRows = [
        ['m1', 'm2', 'm3', ''],
    ];
    let container = document.getElementById("meshContainer2");
    meshRows.forEach((meshIds) => {
        let row = document.createElement("DIV");
        row.classList = "row";

        meshIds.forEach((meshId) => {
            let col = document.createElement("DIV");
            col.classList = "col-md-3 col-sm-3 my-auto";
            if (meshId == "") {
                // Model viewer.
                let model = document.createElement("model-viewer");
                col.appendChild(model);
                row.appendChild(col);
            }
            else {


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
                // controls.appendChild(buttonToggle);

                // Caption.
                let caption = document.createElement("p");
                caption.classList = "caption";
                caption.title = meshAttributes[meshId]["caption"] || "";
                caption.appendChild(document.createTextNode(meshAttributes[meshId]["shortCaption"] || caption.title));

                col.appendChild(model);
                col.appendChild(controls);
                col.appendChild(caption);
                row.appendChild(col);

            }
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
            };
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
