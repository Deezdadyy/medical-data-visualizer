from medical_data_visualizer import draw_cat_plot, draw_heat_map

if __name__ == "__main__":
    fig1 = draw_cat_plot()
    fig1.savefig("catplot.png")

    fig2 = draw_heat_map()
    fig2.savefig("heatmap.png")
