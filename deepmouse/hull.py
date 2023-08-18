import numpy as np


def cross_prod(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - \
           (p1[1] - p0[1]) * (p2[0] - p0[0])


def left_turn(p0, p1, p2):
    return cross_prod(p0, p1, p2) > 0


def convex_hull(points):
    # https://sandipandey.wixsite.com/simplydatascience/post/implementing-a-few-advanced-algorithms-with-python
    n = len(points)
    # the first point
    _, p0 = min([((x, y), i) for i, (x, y) in enumerate(points)])
    start_index = p0
    # print(p0)

    def close_enough(p0, p1):
        distance = ((points[p0][0] - points[p1][0])**2 + (points[p0][1] - points[p1][1])**2)**.5
        return p0 != p1 and distance < 10

    chull = []
    # while (True):
    for i in range(50):
        chull.append(p0)
        p1 = (p0 + 1) % n  # make sure p1 != p0

        for p2 in range(n):
            if left_turn(points[p0], points[p1], points[p2]):
                p1 = p2

        p0 = p1
        # came back to first point?
        if p0 == start_index:
            break

    return chull


def concave_hull(points):
    chull = convex_hull(points)

    def dist(pa, pb):
        return ((points[pa][0] - points[pb][0]) ** 2 + (points[pa][1] - points[pb][1]) ** 2) ** .5

    def between(pa, pb, pc):
        # True if projection of pc onto pa-pb is between pa and pb
        rel_b = np.array(points[pb]) - np.array(points[pa])
        unit_b = rel_b / np.linalg.norm(rel_b)
        rel_c = np.array(points[pc]) - np.array(points[pa])
        scalar_projection = np.dot(rel_c, unit_b)
        return scalar_projection > 0 and scalar_projection < dist(pa, pb)

    def concavity(pa, pb, pc):
        # concavity of curve through pc relative to line from pa to pb
        # note concavity is rate of change of derivative
        ab_distance = dist(pa, pb)
        ab_angle = np.arctan2(points[pb][1]-points[pa][1], points[pb][0]-points[pa][0])
        ac_distance = dist(pa, pc)
        ac_angle = np.arctan2(points[pc][1]-points[pa][1], points[pc][0]-points[pa][0])

        # rotate and scale as if pa and pb are at -1,0 and 1,0
        normalized_angle = ac_angle - ab_angle
        normalized_distance = ac_distance / ab_distance
        x = -1 + 2*normalized_distance*np.cos(normalized_angle)
        y = 2*normalized_distance*np.sin(normalized_angle)

        # solve y = alpha x^2 - alpha for alpha
        return y/(x**2-1) # scale of the parabola on which pc lies

    def next(index):
        if index == len(chull)-1:
            return 0
        else:
            return index+1

    distances = []
    for i in range(len(chull)):
        distances.append(dist(i, next(i)))

    mean_distance = np.mean(distances)

    def find_least_concave_point(pa, pb, max_concavity=1):
        not_in_hull = np.setdiff1d(range(len(points)), chull)

        least_concave_point = None
        least_convavity = None
        for pc in not_in_hull:
            if between(pa, pb, pc):
                c = concavity(pa, pb, pc)
                if c > 0 and c <= max_concavity:
                    if least_concave_point == None or c < least_convavity:
                        least_concave_point = pc
                        least_convavity = c

        return least_concave_point

    def add_concavities():
        # print('len hull {}'.format(len(chull)))
        added_something = False
        i = 0
        while i < len(chull):
            if dist(chull[i], chull[next(i)]) > 2*mean_distance: # consider concave intermediate points
                pc = find_least_concave_point(chull[i], chull[next(i)])
                if pc:
                    added_something = True
                    chull.insert(i+1, pc)
                    i = i + 1
                    # print('*', end='', flush=True)
                # else:
                    # print('.', end='', flush=True)
            i = i + 1
        # print('')
        return added_something

    while add_concavities():
        pass

    return chull


