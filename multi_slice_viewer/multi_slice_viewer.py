import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def previous_slice():
    pass

def next_slice():
    pass

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    #segmentation = ax.segmentation # TODO: uncomment this for seg_viewer!
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    #ax.images[1].set_array(segmentation[ax.index]) # TODO: uncomment this for segmentation!


def next_slice(ax):
    volume = ax.volume
    #segmentation = ax.segmentation
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    #ax.images[1].set_array(segmentation[ax.index])


#sprejeme 3D numpy array in prikaze sliko (s tipkama J in K se pomikas skozi slice)
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index],cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def seg_viewer(volume, segmentation, cmap_ = "Reds"):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.segmentation = segmentation
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index],cmap='gray')
    ax.imshow(segmentation[ax.index], cmap=cmap_, alpha=0.2) # prej bil Reds, za pet lahko jet
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


#prikaže 3D numpy array z večimi slici/odseki slike
def display_numpy(picture):
    fig = plt.figure()
    iter = int(len(picture) /30)
    for num,slice in enumerate(picture):
        if num>=30:
            break
        y = fig.add_subplot(5,6,num+1)

        y.imshow(picture[num*iter], cmap='gray')
    plt.show()
    return