defmodule MyApp.Prediction do
  use GenServer
  alias MyApp.Events

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  def init(_opts) do
    Events.listen_for_event(__MODULE__, :predict)

    {:ok, %{}}
  end

  def handle_info(
        {:predict, %{video: video}}, state
      ) do
        frame = video |> Evision.VideoCapture.read()
        prediction = predict(serving(), frame)
        Phoenix.PubSub.broadcast(MyApp.PubSub, "Prediction", {:prediction, prediction})

    {:noreply, state}
  end

  defp predict(serving, frame) do
    pred_tensor = frame |> Evision.Mat.to_nx() |> Nx.backend_transfer()
    %{predictions: [%{label: label}]} = Nx.Serving.run(serving, pred_tensor)

    label
  end

  defp serving do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/convnext-tiny-224"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "facebook/convnext-tiny-224"})

    Bumblebee.Vision.image_classification(model_info, featurizer,
      top_k: 1,
      compile: [batch_size: 5],
      defn_options: [compiler: EXLA]
    )
  end
end
