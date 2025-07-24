package com.ibosoninnov.instanttrackinglib;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto;
import java.util.ArrayList;
import java.util.Iterator;

/* loaded from: classes2.dex */
public class StickerManager {
    private static int globalIDLimit = 1;
    private float anchorX;
    private float anchorY;
    private Render currentRender;
    private final int stickerId;
    private float userRotation;
    private float userScalingFactor;

    /* loaded from: classes2.dex */
    public enum Render {
        GIF,
        ASSET_3D;

        public Render iterate() {
            values();
            return values()[(ordinal() + 1) % 2];
        }
    }

    public StickerManager() {
        this.userRotation = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.userScalingFactor = 1.0f;
        this.currentRender = Render.values()[1];
        setAnchorCoordinate(0.5f, 0.5f);
        int i = globalIDLimit;
        globalIDLimit = i + 1;
        this.stickerId = i;
    }

    public static StickerBufferProto.StickerRoll getMessageLiteData(ArrayList<StickerManager> arrayList) {
        StickerBufferProto.StickerRoll.Builder newBuilder = StickerBufferProto.StickerRoll.newBuilder();
        Iterator<StickerManager> it = arrayList.iterator();
        while (it.hasNext()) {
            StickerManager next = it.next();
            newBuilder.addSticker(StickerBufferProto.Sticker.newBuilder().setId(next.getstickerId()).setX(next.getAnchorX()).setY(next.getAnchorY()).setRotation(next.getRotation()).setScale(next.getScaleFactor()).setRenderId(next.getRender().ordinal()).build());
        }
        return newBuilder.build();
    }

    public float getAnchorX() {
        return this.anchorX;
    }

    public float getAnchorY() {
        return this.anchorY;
    }

    public Render getRender() {
        return this.currentRender;
    }

    public float getRotation() {
        return this.userRotation;
    }

    public float getScaleFactor() {
        return this.userScalingFactor;
    }

    public int getstickerId() {
        return this.stickerId;
    }

    public void setAnchorCoordinate(float f2, float f3) {
        this.anchorX = f2;
        this.anchorY = f3;
    }

    public void setRender(Render render) {
        this.currentRender = render;
    }

    public void setRotation(float f2) {
        this.userRotation = f2;
    }

    public void setScaleFactor(float f2) {
        this.userScalingFactor = f2;
    }

    public StickerManager(Render render) {
        this.userRotation = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.userScalingFactor = 1.0f;
        this.currentRender = render;
        setAnchorCoordinate(0.5f, 0.5f);
        int i = globalIDLimit;
        globalIDLimit = i + 1;
        this.stickerId = i;
    }
}