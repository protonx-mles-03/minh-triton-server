import numpy as np
import gradio as gr
import client

FLIP_CHOICES = ["origin", "flipped image"]


def flip_image(x, choice):
  """Flip image

  Parameters
  ----------
  x : Image
      Input image
  choice : string
      Flip or not

  Returns
  -------
  Image
      Origin/Flipped image
  """
  if choice == FLIP_CHOICES[1]:
    return np.fliplr(x)
  else:
    return x


"""GUI components for text detection
"""
with gr.Blocks() as gr_client:
  gr.Markdown("**Text Detection**")

  # For image
  with gr.Tab("Image"):
    with gr.Row():
      image_input = gr.Image(label="Upload image for text detection",
                              type="pil",
                              height=500,
                              width=500,
                              )
      infer_image_input = gr.Image( label="Inference image",
                                    type="pil",
                                    height=500,
                                    width=500,
                                  )

    with gr.Row():
      image_choice = gr.Radio(  choices=FLIP_CHOICES,
                                label="Image for inference",
                                info="Original or flipped image?",
                                # value=FLIP_CHOICES[0],
                                )
      img_button = gr.Button("Detect text")
      txt_output = gr.Textbox(label="Detectected text")

      # When input image changes, infer_image_input will change accordingly
      image_input.change( fn=flip_image,
                          inputs=[image_input, image_choice],
                          outputs=infer_image_input)
      # When image_choice changes, infer_image_input will change accordingly
      image_choice.change( fn=flip_image,
                          inputs=[image_input, image_choice],
                          outputs=infer_image_input)
      # Inference when button is clicked
      infer_input = image_input if image_choice == FLIP_CHOICES[0] else infer_image_input      
      img_button.click( fn=client.inference,
                        inputs=infer_input,
                        outputs=txt_output,
                        api_name="inference",
                        )

  with gr.Accordion("For more information:"):
    gr.Markdown("This is a demo for text detection using Triton Inference Server.")
  

if __name__ == "__main__":
  gr_client.launch(share=False)
