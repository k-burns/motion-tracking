defmodule MyAppWeb.HomeLive.Index do
  use MyAppWeb, :live_view
  import Bitwise

  @impl true
  def mount(_params, _, socket) do
    {:ok,
     socket
     |> assign(running?: false)
     |> assign(image: nil)
     |> assign(prediction: nil)
     |> assign(serving: serving())
     |> assign(
       video:
         Evision.VideoCapture.videoCapture(
           "/Users/katelynnburns/Documents/Zoom/2023-07-25 10.51.03 Katelynn (she_her)'s Zoom Meeting/video1536544321.mp4"
         )
     )}
  end

  @impl true

  def handle_event("start", _params, socket) do
    send(self(), :run)

    {:noreply, assign(socket, running?: true)}
  end

  @impl true

  def handle_info(:run, %{assigns: %{running?: true}} = socket) do
    frame = socket.assigns.video |> Evision.VideoCapture.read()

    [predication, image] = predict(socket.assigns.serving, frame)

    send(self(), :run)

    {:noreply,
     socket
     |> assign(prediction: predication)
     |> assign(image: Evision.imencode(".jpg", image) |> Base.encode64())}
  end

  def handle_info(_msg, socket), do: {:noreply, socket}

  defp predict(serving, frame) do
    pred_tensor = frame |> Evision.Mat.to_nx() |> Nx.backend_transfer()
    tensor = frame |> Evision.cvtColor(7) |> Evision.gaussianBlur({7, 7}, 1)
    canny = Evision.canny(tensor, 50, 190)
    kernel = Evision.Mat.ones({3, 3}, :u8)
    di = Evision.dilate(canny, kernel, iterations: 3)
    closed = Evision.morphologyEx(di, Evision.Constant.cv_MORPH_CLOSE(), kernel)

    {contours, _} =
      Evision.findContours(
        closed,
        Evision.Constant.cv_RETR_LIST(),
        Evision.Constant.cv_CHAIN_APPROX_NONE()
      )

    minimal_area = 100

    contours =
      Enum.reject(contours, fn c ->
        # Calculate the area of each contour
        area = Evision.contourArea(c)
        # Ignore contours that are too small or too large
        # (return true to reject)
        area < minimal_area
      end)

    # Enum.map(contours, fn c ->
    #   # area = Evision.contourArea(c)
    #   {x, y, w, h} = Evision.boundingRect(c)
    #   cropped_img=tensor[y: y+h, x: x+w]
    #   img_name= "#{x}.jpg"
    #   Evision.imwrite(img_name, cropped_img)
    # end)

    # IO.inspect("#{Enum.count(contours)} contour(s) remains")

    # color in {Blue, Green, Red}, range from 0-255
    edge_color = {0, 0, 255}

    # # draw all contours by setting `index` to `-1`
    index = -1

    # # Load image in color

    # # draw all contours on the color image
    image = Evision.drawContours(frame, contours, index, edge_color, thickness: 2)
    %{predictions: [%{label: label}]} = Nx.Serving.run(serving, pred_tensor)

    [label, image]
  end

  defp serving do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/convnext-tiny-224"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "facebook/convnext-tiny-224"})

    Bumblebee.Vision.image_classification(model_info, featurizer,
      top_k: 1,
      compile: [batch_size: 1],
      defn_options: [compiler: EXLA]
    )
  end
end
