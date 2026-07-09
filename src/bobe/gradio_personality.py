"""Gradio personality UI components and wiring."""

from __future__ import annotations
from typing import Any

import gradio as gr

from bobe.config import LOCKED_PROFILE, config
from bobe.personality.store import (
    DEFAULT_OPTION,
    sanitize_name,
    write_profile,
    read_voice_for,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
    read_instructions_for,
)


class PersonalityUI:
    """Container for personality-related Gradio components."""

    def __init__(self) -> None:
        """Initialize the PersonalityUI instance."""
        self.DEFAULT_OPTION = DEFAULT_OPTION

        self.personalities_dropdown: gr.Dropdown
        self.apply_btn: gr.Button
        self.status_md: gr.Markdown
        self.preview_md: gr.Markdown
        self.person_name_tb: gr.Textbox
        self.person_instr_ta: gr.TextArea
        self.tools_txt_ta: gr.TextArea
        self.voice_dropdown: gr.Dropdown
        self.new_personality_btn: gr.Button
        self.available_tools_cg: gr.CheckboxGroup
        self.save_btn: gr.Button

    def create_components(self) -> None:
        """Instantiate Gradio components for the personality UI."""
        if LOCKED_PROFILE is not None:
            is_locked = True
            current_value: str = LOCKED_PROFILE
            dropdown_label = "Select personality (locked)"
            dropdown_choices: list[str] = [LOCKED_PROFILE]
        else:
            is_locked = False
            current_value = config.REACHY_MINI_CUSTOM_PROFILE or self.DEFAULT_OPTION
            dropdown_label = "Select personality"
            dropdown_choices = [self.DEFAULT_OPTION, *list_personalities()]

        self.personalities_dropdown = gr.Dropdown(
            label=dropdown_label,
            choices=dropdown_choices,
            value=current_value,
            interactive=not is_locked,
        )
        self.apply_btn = gr.Button("Apply personality", interactive=not is_locked)
        self.status_md = gr.Markdown(visible=True)
        self.preview_md = gr.Markdown(value=read_instructions_for(current_value))
        self.person_name_tb = gr.Textbox(label="Personality name", interactive=not is_locked)
        self.person_instr_ta = gr.TextArea(label="Personality instructions", lines=10, interactive=not is_locked)
        self.tools_txt_ta = gr.TextArea(label="tools.txt", lines=10, interactive=not is_locked)
        self.voice_dropdown = gr.Dropdown(label="Voice", choices=["cedar"], value="cedar", interactive=not is_locked)
        self.new_personality_btn = gr.Button("New personality", interactive=not is_locked)
        self.available_tools_cg = gr.CheckboxGroup(
            label="Available tools (helper)",
            choices=[],
            value=[],
            interactive=not is_locked,
        )
        self.save_btn = gr.Button("Save personality (instructions + tools)", interactive=not is_locked)

    def additional_inputs_ordered(self) -> list[Any]:
        """Return the additional inputs in the expected order for Stream."""
        return [
            self.personalities_dropdown,
            self.apply_btn,
            self.new_personality_btn,
            self.status_md,
            self.preview_md,
            self.person_name_tb,
            self.person_instr_ta,
            self.tools_txt_ta,
            self.voice_dropdown,
            self.available_tools_cg,
            self.save_btn,
        ]

    def wire_events(self, handler: Any, blocks: gr.Blocks) -> None:
        """Attach event handlers to components within a Blocks context."""

        async def _apply_personality(selected: str) -> tuple[str, str]:
            if LOCKED_PROFILE is not None and selected != LOCKED_PROFILE:
                return (
                    f"Profile is locked to '{LOCKED_PROFILE}'. Cannot change personality.",
                    read_instructions_for(LOCKED_PROFILE),
                )
            profile = None if selected == self.DEFAULT_OPTION else selected
            status = await handler.apply_personality(profile)
            preview = read_instructions_for(selected)
            return status, preview

        async def _fetch_voices(selected: str) -> dict[str, Any]:
            try:
                voices = await handler.get_available_voices()
                current = read_voice_for(selected)
                if current not in voices:
                    current = "cedar"
                return gr.update(choices=voices, value=current)
            except Exception:
                return gr.update(choices=["cedar"], value="cedar")

        def _parse_enabled_tools(text: str) -> list[str]:
            enabled: list[str] = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                enabled.append(stripped)
            return enabled

        def _load_profile_for_edit(selected: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
            instr = read_instructions_for(selected)
            tools_txt = ""
            if selected != self.DEFAULT_OPTION:
                tools_path = resolve_profile_dir(selected) / "tools.txt"
                if tools_path.exists():
                    tools_txt = tools_path.read_text(encoding="utf-8")
            all_tools = available_tools_for(selected)
            enabled = _parse_enabled_tools(tools_txt)
            status_text = f"Loaded profile '{selected}'."
            return (
                gr.update(value=instr),
                gr.update(value=tools_txt),
                gr.update(choices=all_tools, value=enabled),
                status_text,
            )

        def _new_personality() -> tuple[
            dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]
        ]:
            try:
                instr_val = "# Write your instructions here\n# e.g., Keep responses concise and friendly."
                tools_txt_val = "# tools enabled for this profile\n"
                return (
                    gr.update(value=""),
                    gr.update(value=instr_val),
                    gr.update(value=tools_txt_val),
                    gr.update(choices=available_tools_for(self.DEFAULT_OPTION), value=[]),
                    "Fill in a name, instructions and (optional) tools, then Save.",
                    gr.update(value="cedar"),
                )
            except Exception:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "Failed to initialize new personality.",
                    gr.update(),
                )

        def _save_personality(
            name: str, instructions: str, tools_text: str, voice: str
        ) -> tuple[dict[str, Any], dict[str, Any], str]:
            if not sanitize_name(name):
                return gr.update(), gr.update(), "Please enter a valid name."
            try:
                value = write_profile(name, instructions, tools_text, voice or "cedar")
                choices = [self.DEFAULT_OPTION, *sorted({*list_personalities(), value})]
                return (
                    gr.update(choices=choices, value=value),
                    gr.update(value=instructions),
                    f"Saved personality '{sanitize_name(name)}'.",
                )
            except Exception as exc:
                return gr.update(), gr.update(), f"Failed to save personality: {exc}"

        def _sync_tools_from_checks(selected: list[str], current_text: str) -> dict[str, Any]:
            comments = [line for line in current_text.splitlines() if line.strip().startswith("#")]
            body = "\n".join(selected)
            out = ("\n".join(comments) + ("\n" if comments else "") + body).strip() + "\n"
            return gr.update(value=out)

        with blocks:
            self.apply_btn.click(
                fn=_apply_personality,
                inputs=[self.personalities_dropdown],
                outputs=[self.status_md, self.preview_md],
            )

            self.personalities_dropdown.change(
                fn=_load_profile_for_edit,
                inputs=[self.personalities_dropdown],
                outputs=[self.person_instr_ta, self.tools_txt_ta, self.available_tools_cg, self.status_md],
            )

            blocks.load(
                fn=_fetch_voices,
                inputs=[self.personalities_dropdown],
                outputs=[self.voice_dropdown],
            )

            self.available_tools_cg.change(
                fn=_sync_tools_from_checks,
                inputs=[self.available_tools_cg, self.tools_txt_ta],
                outputs=[self.tools_txt_ta],
            )

            self.new_personality_btn.click(
                fn=_new_personality,
                inputs=[],
                outputs=[
                    self.person_name_tb,
                    self.person_instr_ta,
                    self.tools_txt_ta,
                    self.available_tools_cg,
                    self.status_md,
                    self.voice_dropdown,
                ],
            )

            self.save_btn.click(
                fn=_save_personality,
                inputs=[self.person_name_tb, self.person_instr_ta, self.tools_txt_ta, self.voice_dropdown],
                outputs=[self.personalities_dropdown, self.person_instr_ta, self.status_md],
            ).then(
                fn=_apply_personality,
                inputs=[self.personalities_dropdown],
                outputs=[self.status_md, self.preview_md],
            )
