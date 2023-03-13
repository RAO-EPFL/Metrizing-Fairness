__author__ = "Xinyue"

from dccp.linearize import linearize, linearize_para
import cvxpy as cvx

# from dccp.linearize import linearize_para
def convexify_para_constr(self):
    """
    input:
        self: a constraint of a problem
    return:
        if the constraint is dcp, return itself;
        otherwise, return
            a convexified constraint
            para: [left side, right side]
                if the left/right-hand side of the the constraint is linearized,
                left/right side = [zero order parameter, {variable: [value parameter, [gradient parameter]]}]
                else,
                left/right side = []
            dom: domain
    """
    if not self.is_dcp():
        dom = []  # domain
        para = []  # a list for parameters
        if self.expr.args[0].curvature == "CONCAVE":  # left-hand concave
            lin = linearize_para(self.expr.args[0])  # linearize the expression
            left = lin[0]
            para.append(
                [lin[1], lin[2]]
            )  # [zero order parameter, {variable: [value parameter, [gradient parameter]]}]
            for con in lin[3]:
                dom.append(con)
        else:
            left = self.expr.args[0]
            para.append(
                []
            )  # appending an empty list indicates the expression has the right curvature
        if (
            self.expr.args[1].curvature == "CONCAVE"
        ):  # negative right-hand must be concave (right-hand is convex)
            lin = linearize_para(self.expr.args[1])  # linearize the expression
            neg_right = lin[0]
            para.append([lin[1], lin[2]])
            for con in lin[3]:
                dom.append(con)
        else:
            neg_right = self.expr.args[1]
            para.append([])
        return left + neg_right <= 0, para, dom
    else:
        return self


def convexify_constr(constr):
    """
    :param constr: a constraint of a problem
    :return:
    for a dcp constraint, return itself;
    for a non-dcp constraint, return a convexified constraint and domain constraints;
    return None if non-sub/super-diff
    """
    if not constr.is_dcp():
        dom = []
        # left hand concave
        if constr.args[0].curvature == "CONCAVE":
            left = linearize(constr.args[0])
            if left is None:
                return None
            else:
                for con in constr.args[0].domain:
                    dom.append(con)
        else:
            left = constr.args[0]
        # right hand convex
        if constr.args[1].curvature == "CONVEX":
            right = linearize(constr.args[1])
            if right is None:
                return None
            else:
                for con in constr.args[1].domain:
                    dom.append(con)
        else:
            right = constr.args[1]
        return left - right <= 0, dom
    else:
        return constr
