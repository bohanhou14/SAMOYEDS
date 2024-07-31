import matplotlib.pyplot as plt
import numpy as np
labels = ['female', 'male']
# data = [443, 57]
data = [467, 30]

# labels = ['White', 'African American', 'Asian', 'Hispanic', "Middle Eastern", "Native American"]
# data = [261, 116, 96, 19, 1, 0]
# labels = ['less than high school', 'some high school', 'high school', 'some college', 'college', 'postgraduate']
# data = [17, 32, 101, 96, 210, 1] US
# labels = ['high school', 'some college', 'college', 'postgraduate']
# data = [16, 96, 387, 1]



if __name__ == '__main__':
    x = np.char.array(labels)
    y = np.array(data)
    colors = ['lightskyblue', 'violet', 'blue', 'red', 'green']
    porcent = 100. * y / y.sum()

    patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, porcent)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y),
                                             key=lambda x: x[2],
                                             reverse=True))

    plt.legend(patches, labels, loc='lower center', bbox_to_anchor=(-0.1, 1.),
               fontsize=7)

    state = "US"
    # state = "Maryland"

    # attr = "race"
    attr = "gender"
    # attr = "education"
    atti = "approval"
    title = f"{state}-500-{atti}-{attr}"
    plt.title(title)
    plt.savefig(f"fig/{title}",  bbox_inches='tight')