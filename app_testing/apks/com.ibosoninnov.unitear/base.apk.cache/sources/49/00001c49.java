package com.google.ar.sceneform;

import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;

/* loaded from: classes.dex */
public abstract class NodeParent {
    private final ArrayList<Node> children;
    private boolean isIterableChildrenDirty;
    private final ArrayList<Node> iterableChildren;
    private int iteratingCounter;
    private final List<Node> unmodifiableChildren;

    public NodeParent() {
        ArrayList<Node> arrayList = new ArrayList<>();
        this.children = arrayList;
        this.unmodifiableChildren = Collections.unmodifiableList(arrayList);
        this.iterableChildren = new ArrayList<>();
    }

    private ArrayList<Node> getIterableChildren() {
        if (this.isIterableChildrenDirty && !isIterating()) {
            this.iterableChildren.clear();
            this.iterableChildren.addAll(this.children);
            this.isIterableChildrenDirty = false;
        }
        return this.iterableChildren;
    }

    private boolean isIterating() {
        return this.iteratingCounter > 0;
    }

    private void startIterating() {
        this.iteratingCounter++;
    }

    private void stopIterating() {
        int i = this.iteratingCounter - 1;
        this.iteratingCounter = i;
        if (i < 0) {
            throw new AssertionError("stopIteration was called without calling startIteration.");
        }
    }

    public final void addChild(Node node) {
        Preconditions.checkNotNull(node, "Parameter \"child\" was null.");
        AndroidPreconditions.checkUiThread();
        if (node.parent == this) {
            return;
        }
        StringBuilder sb = new StringBuilder();
        if (canAddChild(node, sb)) {
            onAddChild(node);
            return;
        }
        throw new IllegalArgumentException(sb.toString());
    }

    public void callOnHierarchy(Consumer<Node> consumer) {
        Preconditions.checkNotNull(consumer, "Parameter \"consumer\" was null.");
        ArrayList<Node> iterableChildren = getIterableChildren();
        startIterating();
        for (int i = 0; i < iterableChildren.size(); i++) {
            iterableChildren.get(i).callOnHierarchy(consumer);
        }
        stopIterating();
    }

    public boolean canAddChild(Node node, StringBuilder sb) {
        Preconditions.checkNotNull(node, "Parameter \"child\" was null.");
        Preconditions.checkNotNull(sb, "Parameter \"failureReason\" was null.");
        if (node == this) {
            sb.append("Cannot add child: Cannot make a node a child of itself.");
            return false;
        }
        return true;
    }

    public Node findByName(final String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        final int hashCode = str.hashCode();
        return findInHierarchy(new Predicate() { // from class: c.d.b.a.g
            @Override // java.util.function.Predicate
            public final boolean test(Object obj) {
                int i = hashCode;
                String str2 = str;
                Node node = (Node) obj;
                String name = node.getName();
                return (node.getNameHash() != 0 && node.getNameHash() == i) || (name != null && name.equals(str2));
            }
        });
    }

    public Node findInHierarchy(Predicate<Node> predicate) {
        Preconditions.checkNotNull(predicate, "Parameter \"condition\" was null.");
        ArrayList<Node> iterableChildren = getIterableChildren();
        startIterating();
        Node node = null;
        for (int i = 0; i < iterableChildren.size() && (node = iterableChildren.get(i).findInHierarchy(predicate)) == null; i++) {
        }
        stopIterating();
        return node;
    }

    public final List<Node> getChildren() {
        return this.unmodifiableChildren;
    }

    public void onAddChild(Node node) {
        Preconditions.checkNotNull(node, "Parameter \"child\" was null.");
        NodeParent nodeParent = node.getNodeParent();
        if (nodeParent != null) {
            nodeParent.removeChild(node);
        }
        this.children.add(node);
        node.parent = this;
        this.isIterableChildrenDirty = true;
    }

    public void onRemoveChild(Node node) {
        Preconditions.checkNotNull(node, "Parameter \"child\" was null.");
        this.children.remove(node);
        node.parent = null;
        this.isIterableChildrenDirty = true;
    }

    public final void removeChild(Node node) {
        Preconditions.checkNotNull(node, "Parameter \"child\" was null.");
        AndroidPreconditions.checkUiThread();
        if (this.children.contains(node)) {
            onRemoveChild(node);
        }
    }
}