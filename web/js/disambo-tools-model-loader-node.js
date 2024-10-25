import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

// PresetText.js content

let replaceRegex;
const id = "pysssss.PresetText.Presets";
const MISSING = Symbol();

const getPresets = () => {
	let items;
	try {
		items = JSON.parse(localStorage.getItem(id));
	} catch (error) {}
	if (!items || !items.length) {
		items = [{ name: "default negative", value: "worst quality" }];
	}
	return items;
};

let presets = getPresets();

app.registerExtension({
	name: "pysssss.PresetText",
	setup() {
		app.ui.settings.addSetting({
			id: "pysssss.PresetText.ReplacementRegex",
			name: "🐍 Preset Text Replacement Regex",
			type: "text",
			defaultValue: "(?:^|[^\\w])(?<replace>@(?<id>[\\w-]+))",
			tooltip:
				"The regex should return two named capture groups: id (the name of the preset text to use), replace (the matched text to replace)",
			attrs: {
				style: {
					fontFamily: "monospace",
				},
			},
			onChange(value) {
				if (!value) {
					replaceRegex = null;
					return;
				}
				try {
					replaceRegex = new RegExp(value, "g");
				} catch (error) {
					alert("Error creating regex for preset text replacement, no replacements will be performed.");
					replaceRegex = null;
				}
			},
		});

		const drawNodeWidgets = LGraphCanvas.prototype.drawNodeWidgets
		LGraphCanvas.prototype.drawNodeWidgets = function(node) {
			const c = LiteGraph.WIDGET_BGCOLOR;
			try {
				if(node[MISSING]) {
					LiteGraph.WIDGET_BGCOLOR = "red";
				}
				return drawNodeWidgets.apply(this, arguments);
			} finally {
				LiteGraph.WIDGET_BGCOLOR = c;
			}
		};
	},
	registerCustomNodes() {
		class PresetTextNode {
			constructor() {
				this.isVirtualNode = true;
				this.serialize_widgets = true;
				this.addOutput("text", "STRING");

				const widget = this.addWidget("combo", "value", presets[0].name, () => {}, {
					values: presets.map((p) => p.name),
				});
				this.addWidget("button", "Manage", "Manage", () => {
					const container = document.createElement("div");
					Object.assign(container.style, {
						display: "grid",
						gridTemplateColumns: "1fr 1fr",
						gap: "10px",
					});

					const addNew = document.createElement("button");
					addNew.textContent = "Add New";
					addNew.classList.add("pysssss-presettext-addnew");
					Object.assign(addNew.style, {
						fontSize: "13px",
						gridColumn: "1 / 3",
						color: "dodgerblue",
						width: "auto",
						textAlign: "center",
					});
					addNew.onclick = () => {
						addRow({ name: "", value: "" });
					};
					container.append(addNew);

					function addRow(p) {
						const name = document.createElement("input");
						const nameLbl = document.createElement("label");
						name.value = p.name;
						nameLbl.textContent = "Name:";
						nameLbl.append(name);

						const value = document.createElement("input");
						const valueLbl = document.createElement("label");
						value.value = p.value;
						valueLbl.append(value);

						addNew.before(nameLbl, valueLbl);
					}
					for (const p of presets) {
						addRow(p);
					}

					const help = document.createElement("span");
					help.textContent = "To remove a preset set the name or value to blank";
					help.style.gridColumn = "1 / 3";
					container.append(help);

					dialog.show("");
					dialog.textElement.append(container);
				});

				const dialog = new app.ui.dialog.constructor();
				dialog.element.classList.add("comfy-settings");

				const closeButton = dialog.element.querySelector("button");
				closeButton.textContent = "CANCEL";
				const saveButton = document.createElement("button");
				saveButton.textContent = "SAVE";
				saveButton.onclick = function () {
					const inputs = dialog.element.querySelectorAll("input");
					const p = [];
					for (let i = 0; i < inputs.length; i += 2) {
						const n = inputs[i];
						const v = inputs[i + 1];
						if (!n.value.trim() || !v.value.trim()) {
							continue;
						}
						p.push({ name: n.value, value: v.value });
					}

					widget.options.values = p.map((p) => p.name);
					if (!widget.options.values.includes(widget.value)) {
						widget.value = widget.options.values[0];
					}

					presets = p;
					localStorage.setItem(id, JSON.stringify(presets));

					dialog.close();
				};

				closeButton.before(saveButton);

				this.applyToGraph = function (workflow) {
					// For each output link copy our value over the original widget value
					if (this.outputs[0].links && this.outputs[0].links.length) {
						for (const l of this.outputs[0].links) {
							const link_info = app.graph.links[l];
							const outNode = app.graph.getNodeById(link_info.target_id);
							const outIn = outNode && outNode.inputs && outNode.inputs[link_info.target_slot];
							if (outIn.widget) {
								const w = outNode.widgets.find((w) => w.name === outIn.widget.name);
								if (!w) continue;
								const preset = presets.find((p) => p.name === widget.value);
								if (!preset) {
									this[MISSING] = true;
									app.graph.setDirtyCanvas(true, true);
									const msg = `Preset text '${widget.value}' not found. Please fix this and queue again.`;
									throw new Error(msg);
								}
								delete this[MISSING];
								w.value = preset.value;
							}
						}
					}
				};
			}
		}

		LiteGraph.registerNodeType(
			"PresetText|pysssss",
			Object.assign(PresetTextNode, {
				title: "Preset Text 🐍",
			})
		);

		PresetTextNode.category = "utils";
	},
	nodeCreated(node) {
		if (node.widgets) {
			// Locate dynamic prompt text widgets
			const widgets = node.widgets.filter((n) => n.type === "customtext" || n.type === "text");
			for (const widget of widgets) {
				const callbacks = [
					() => {
						let prompt = widget.value;
						if (replaceRegex && typeof prompt.replace !== 'undefined') {
							prompt = prompt.replace(replaceRegex, (match, p1, p2, index, text, groups) => {
								if (!groups.replace || !groups.id) return match; // No match, bad regex?

								const preset = presets.find((p) => p.name.replaceAll(/\s/g, "-") === groups.id);
								if (!preset) return match; // Invalid name

								const pos = match.indexOf(groups.replace);
								return match.substring(0, pos) + preset.value;
							});
						}
						return prompt;
					},
				];
				let inheritedSerializeValue = widget.serializeValue || null;

				let called = false;
				const serializeValue = async (workflowNode, widgetIndex) => {
					const origWidgetValue = widget.value;
					if (called) return origWidgetValue;
					called = true;

					let allCallbacks = [...callbacks];
					if (inheritedSerializeValue) {
						allCallbacks.push(inheritedSerializeValue);
					}
					let valueIsUndefined = false;

					for (const cb of allCallbacks) {
						let value = await cb(workflowNode, widgetIndex);
						// Need to check the callback return value before it is set on widget.value as it coerces it to a string (even for undefined)
						if (value === undefined) valueIsUndefined = true;
						widget.value = value;
					}

					const prompt = valueIsUndefined ? undefined : widget.value;
					widget.value = origWidgetValue;

					called = false;

					return prompt;
				};

				Object.defineProperty(widget, "serializeValue", {
					get() {
						return serializeValue;
					},
					set(cb) {
						inheritedSerializeValue = cb;
					},
				});
			}
		}
	},
});

// betterCombos.js content

const CHECKPOINT_LOADER = "CheckpointLoaderNode";
const LORA_LOADER = "LoraLoaderNode";

function getType(node) {
	if (node.comfyClass === CHECKPOINT_LOADER) {
		return "checkpoints";
	}
	return "loras";
}

app.registerExtension({
	name: "pysssss.Combo++",
	init() {
		$el("style", {
			textContent: `
				.litemenu-entry:hover .pysssss-combo-image {
					display: block;
				}
				.pysssss-combo-image {
					display: none;
					position: absolute;
					left: 0;
					top: 0;
					transform: translate(-100%, 0);
					width: 384px;
					height: 384px;
					background-size: contain;
					background-position: top right;
					background-repeat: no-repeat;
					filter: brightness(65%);
				}
			`,
			parent: document.body,
		});

		const submenuSetting = app.ui.settings.addSetting({
			id: "pysssss.Combo++.Submenu",
			name: "🐍 Enable submenu in custom nodes",
			defaultValue: true,
			type: "boolean",
		});

		// Hook and add callback mechanisms
		const getOrSet = (target, name, create) => {
			if (name in target) return target[name];
			return (target[name] = create());
		};
		const symbol = getOrSet(window, "__pysssss__", () => Symbol("__pysssss__"));
		const store = getOrSet(window, symbol, () => ({}));
		const contextMenuHook = getOrSet(store, "contextMenuHook", () => ({}));
		for (const e of ["ctor", "preAddItem", "addItem"]) {
			if (!contextMenuHook[e]) {
				contextMenuHook[e] = [];
			}
		}

		// Checks if this is a custom combo item
		const isCustomItem = (value) => value && typeof value === "object" && "image" in value && value.content;
		const splitBy = (navigator.platform || navigator.userAgent).includes("Win") ? /\/|\\/ : /\//;

		contextMenuHook["ctor"].push(function (values, options) {
			if (options.parentMenu?.options?.className === "dark") {
				options.className = "dark";
			}
		});

		function encodeRFC3986URIComponent(str) {
			return encodeURIComponent(str).replace(/[!'()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
		}

		// After an element is created for an item, add an image if it has one
		contextMenuHook["addItem"].push(function (el, menu, [name, value, options]) {
			if (el && isCustomItem(value) && value?.image && !value.submenu) {
				el.textContent += " *";
				$el("div.pysssss-combo-image", {
					parent: el,
					style: {
						backgroundImage: `url(/pysssss/view/${encodeRFC3986URIComponent(value.image)})`,
					},
				});
			}
		});

		function buildMenu(widget, values) {
			const lookup = {
				"": { options: [] },
			};

			// Split paths into menu structure
			for (const value of values) {
				const split = value.content.split(splitBy);
				let path = "";
				for (let i = 0; i < split.length; i++) {
					const s = split[i];
					const last = i === split.length - 1;
					if (last) {
						lookup[path].options.push({
							...value,
							title: s,
							callback: () => {
								widget.value = value;
								widget.callback(value);
								app.graph.setDirtyCanvas(true);
							},
						});
					} else {
						const prevPath = path;
						path += s + splitBy;
						if (!lookup[path]) {
							const sub = {
								title: s,
								submenu: {
									options: [],
									title: s,
								},
							};
							lookup[path] = sub.submenu;
							lookup[prevPath].options.push(sub);
						}
					}
				}
			}

			return lookup[""].options;
		}

		// Override COMBO widgets to patch their values
		const combo = ComfyWidgets["COMBO"];
		ComfyWidgets["COMBO"] = function (node, inputName, inputData) {
			const type = inputData[0];
			const res = combo.apply(this, arguments);
			if (isCustomItem(type[0])) {
				let value = res.widget.value;
				let values = res.widget.options.values;
				let menu = null;

				// Override the option values to check if we should render a menu structure
				Object.defineProperty(res.widget.options, "values", {
					get() {
						let v = values;

						if (submenuSetting.value) {
							if (!menu) {
								// Only build the menu once
								menu = buildMenu(res.widget, values);
							}
							v = menu;
						}

						const valuesIncludes = v.includes;
						v.includes = function (searchElement) {
							const includesFromMenuItems = function (items) {
								for (const item of items) {
									if (includesFromMenuItem(item)) {
										return true;
									}
								}
								return false;
							};
							const includesFromMenuItem = function (item) {
								if (item.submenu) {
									return includesFromMenuItems(item.submenu.options);
								} else {
									return item.content === searchElement.content;
								}
							};

							return valuesIncludes.apply(this, arguments) || includesFromMenuItems(this);
						};

						return v;
					},
					set(v) {
						values = v;
						menu = null;
					},
				});

				Object.defineProperty(res.widget, "value", {
					get() {
						if (res.widget) {
							const stack = new Error().stack;
							if (stack.includes("drawNodeWidgets") || stack.includes("saveImageExtraOutput")) {
								return (value || type[0]).content;
							}
						}
						return value;
					},
					set(v) {
						if (v?.submenu) {
							return;
						}
						value = v;
					},
				});
			}

			return res;
		};
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const isCkpt = nodeType.comfyClass === CHECKPOINT_LOADER;
		const isLora = nodeType.comfyClass === LORA_LOADER;
		if (isCkpt || isLora) {
			const onAdded = nodeType.prototype.onAdded;
			nodeType.prototype.onAdded = function () {
				onAdded?.apply(this, arguments);
				const { widget: exampleList } = ComfyWidgets["COMBO"](this, "example", [[""]], app);

				let exampleWidget;

				const get = async (route, suffix) => {
					const url = encodeURIComponent(`${getType(nodeType)}${suffix || ""}`);
					return await api.fetchApi(`/pysssss/${route}/${url}`);
				};

				const getExample = async () => {
					if (exampleList.value === "[none]") {
						if (exampleWidget) {
							exampleWidget.inputEl.remove();
							exampleWidget = null;
							this.widgets.length -= 1;
						}
						return;
					}

					const v = this.widgets[0].value.content;
					const pos = v.lastIndexOf(".");
					const name = v.substr(0, pos);
					let exampleName = exampleList.value;
					let viewPath = `/${name}`;
					if (exampleName === "notes") {
						viewPath += ".txt";
					} else {
						viewPath += `/${exampleName}`;
					}
					const example = await (await get("view", viewPath)).text();
					if (!exampleWidget) {
						exampleWidget = ComfyWidgets["STRING"](this, "prompt", ["STRING", { multiline: true }], app).widget;
						exampleWidget.inputEl.readOnly = true;
						exampleWidget.inputEl.style.opacity = 0.6;
					}
					exampleWidget.value = example;
				};

				const exampleCb = exampleList.callback;
				exampleList.callback = function () {
					getExample();
					return exampleCb?.apply(this, arguments) ?? exampleList.value;
				};

				const listExamples = async () => {
					exampleList.disabled = true;
					exampleList.options.values = ["[none]"];
					exampleList.value = "[none]";
					let examples = [];
					if (this.widgets[0].value?.content) {
						try {
							examples = await (await get("examples", `/${this.widgets[0].value.content}`)).json();
						} catch (error) {}
					}
					exampleList.options.values = ["[none]", ...examples];
					exampleList.value = exampleList.options.values[+!!examples.length];
					exampleList.callback();
					exampleList.disabled = !examples.length;
					app.graph.setDirtyCanvas(true, true);
				};

				// Expose function to update examples
				nodeType.prototype["pysssss.updateExamples"] = listExamples;

				const modelWidget = this.widgets[0];
				const modelCb = modelWidget.callback;
				let prev = undefined;
				modelWidget.callback = function () {
					const ret = modelCb?.apply(this, arguments) ?? modelWidget.value;
					let v = ret;
					if (ret?.content) {
						v = ret.content;
					}
					if (prev !== v) {
						listExamples();
						prev = v;
					}
					return ret;
				};
				setTimeout(() => {
					modelWidget.callback();
				}, 30);
			};

			const addInput = nodeType.prototype.addInput ?? LGraphNode.prototype.addInput;
			nodeType.prototype.addInput = function (_, type) {
				if (type === "HIDDEN") return;
				return addInput.apply(this, arguments);
			};
		}

		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
			if (this.imgs) {
				let img;
				if (this.imageIndex != null) {
					img = this.imgs[this.imageIndex];
				} else if (this.overIndex != null) {
					img = this.imgs[this.overIndex];
				}
				if (img) {
					const nodes = app.graph._nodes.filter(
						(n) => n.comfyClass === LORA_LOADER || n.comfyClass === CHECKPOINT_LOADER
					);
					if (nodes.length) {
						options.unshift({
							content: "Save as Preview",
							submenu: {
								options: nodes.map((n) => ({
									content: n.widgets[0].value.content,
									callback: async () => {
										const url = new URL(img.src);
										const { image } = await api.fetchApi(
											"/pysssss/save/" + encodeURIComponent(`${getType(n)}/${n.widgets[0].value.content}`),
											{
												method: "POST",
												body: JSON.stringify({
													filename: url.searchParams.get("filename"),
													subfolder: url.searchParams.get("subfolder"),
													type: url.searchParams.get("type"),
												}),
												headers: {
													"content-type": "application/json",
												},
											}
										);
										n.widgets[0].value.image = image;
										app.refreshComboInNodes();
									},
								})),
							},
						});
					}
				}
			}
			return getExtraMenuOptions?.apply(this, arguments);
		};
	},
});
