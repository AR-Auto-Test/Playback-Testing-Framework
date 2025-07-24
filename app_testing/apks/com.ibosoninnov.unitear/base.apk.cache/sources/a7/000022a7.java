package com.google.common.flogger;

/* loaded from: classes.dex */
public enum StackSize {
    SMALL(10),
    MEDIUM(20),
    LARGE(50),
    FULL(-1),
    NONE(0);
    
    private final int maxDepth;

    StackSize(int i) {
        this.maxDepth = i;
    }

    public int getMaxDepth() {
        return this.maxDepth;
    }
}