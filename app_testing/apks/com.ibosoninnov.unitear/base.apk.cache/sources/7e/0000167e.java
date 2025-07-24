package com.google.android.gms.vision.text;

import android.graphics.Point;
import android.graphics.Rect;
import com.google.android.gms.internal.vision.zzab;

/* compiled from: com.google.android.gms:play-services-vision@@20.1.3 */
/* loaded from: classes.dex */
public final class zzc {
    public static Rect zza(Text text) {
        Point[] cornerPoints;
        int i = Integer.MIN_VALUE;
        int i2 = Integer.MAX_VALUE;
        int i3 = Integer.MIN_VALUE;
        int i4 = Integer.MAX_VALUE;
        for (Point point : text.getCornerPoints()) {
            i4 = Math.min(i4, point.x);
            i = Math.max(i, point.x);
            i2 = Math.min(i2, point.y);
            i3 = Math.max(i3, point.y);
        }
        return new Rect(i4, i2, i, i3);
    }

    public static Point[] zza(zzab zzabVar) {
        double sin = Math.sin(Math.toRadians(zzabVar.zze));
        double cos = Math.cos(Math.toRadians(zzabVar.zze));
        int i = zzabVar.zzc;
        int i2 = (int) ((i * cos) + zzabVar.zza);
        double d2 = zzabVar.zzb;
        int i3 = zzabVar.zzd;
        Point[] pointArr = {new Point(zzabVar.zza, zzabVar.zzb), new Point(i2, (int) ((i * sin) + d2)), new Point((int) (pointArr[1].x - (i3 * sin)), (int) ((i3 * cos) + pointArr[1].y)), new Point((pointArr[2].x - pointArr[1].x) + pointArr[0].x, (pointArr[2].y - pointArr[1].y) + pointArr[0].y)};
        return pointArr;
    }
}