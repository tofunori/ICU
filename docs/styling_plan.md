# Questionnaire Styling Plan (Google Forms - Green Theme)

This plan outlines the steps to restyle the existing questionnaire (`templates/index.html` and `static/style.css`) to resemble the look and feel of a Google Form, using a green theme based on the provided example image.

## Steps

1.  **Foundation:**
    *   Set the main page background to a light grey (e.g., `#f8f9fa`).
    *   Define the primary theme color as green (e.g., `#1e8e3e`).
    *   Use a clean sans-serif font (e.g., Arial, Roboto, or existing `sans-serif`).
2.  **Header Card:**
    *   Restyle the introduction `div` to have a white background, padding, rounded corners, a solid green top border, and a subtle bottom border/shadow.
3.  **Question Cards:**
    *   Style each `section` element as a separate white card with padding, rounded corners, a standard grey border (e.g., `1px solid #dadce0`), and appropriate bottom margin.
4.  **Form Element Styling:**
    *   **Inputs (text, textarea, select):** Style with clean lines, potentially a bottom border that activates on focus.
    *   **Radio Buttons & Checkboxes:** Replicate the style from the example image (outlined circles/squares), hiding the default browser appearance.
    *   **Labels:** Ensure clear, readable font styling.
5.  **Button Styling:**
    *   Style the submit button with the green theme color background, white text, and appropriate padding/border-radius.
6.  **Typography and Spacing:**
    *   Adjust overall spacing (margins, padding) for visual consistency with the example form.

## Visualization

```mermaid
graph TD
    A[Start: Current Style] --> B(Foundation: Set Light Grey BG, Green Theme, Font);
    B --> C(Header Card: Style Intro Div w/ Green Top Border);
    C --> D(Question Cards: Style Sections as White Cards w/ Grey Borders);
    D --> E(Inputs: Style Text, Select, Textarea);
    E --> F(Radios/Checkboxes: Style like Example Image);
    F --> G(Button: Style Submit Button w/ Green Theme);
    G --> H(Refine: Adjust Spacing & Typography);
    H --> I(End: Google Form-like Style - Green Theme);