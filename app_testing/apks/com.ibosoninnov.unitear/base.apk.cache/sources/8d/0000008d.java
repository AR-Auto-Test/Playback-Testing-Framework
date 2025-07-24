package androidx.core.graphics.drawable;

import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.os.Parcelable;
import b.b0.a;
import java.nio.charset.Charset;
import java.util.Objects;

/* loaded from: classes.dex */
public class IconCompatParcelizer {
    public static IconCompat read(a aVar) {
        IconCompat iconCompat = new IconCompat();
        iconCompat.f239b = aVar.k(iconCompat.f239b, 1);
        byte[] bArr = iconCompat.f241d;
        if (aVar.i(2)) {
            bArr = aVar.g();
        }
        iconCompat.f241d = bArr;
        iconCompat.f242e = aVar.m(iconCompat.f242e, 3);
        iconCompat.f243f = aVar.k(iconCompat.f243f, 4);
        iconCompat.f244g = aVar.k(iconCompat.f244g, 5);
        iconCompat.f245h = (ColorStateList) aVar.m(iconCompat.f245h, 6);
        String str = iconCompat.j;
        if (aVar.i(7)) {
            str = aVar.n();
        }
        iconCompat.j = str;
        String str2 = iconCompat.k;
        if (aVar.i(8)) {
            str2 = aVar.n();
        }
        iconCompat.k = str2;
        iconCompat.i = PorterDuff.Mode.valueOf(iconCompat.j);
        switch (iconCompat.f239b) {
            case -1:
                Parcelable parcelable = iconCompat.f242e;
                if (parcelable != null) {
                    iconCompat.f240c = parcelable;
                    break;
                } else {
                    throw new IllegalArgumentException("Invalid icon");
                }
            case 1:
            case 5:
                Parcelable parcelable2 = iconCompat.f242e;
                if (parcelable2 != null) {
                    iconCompat.f240c = parcelable2;
                    break;
                } else {
                    byte[] bArr2 = iconCompat.f241d;
                    iconCompat.f240c = bArr2;
                    iconCompat.f239b = 3;
                    iconCompat.f243f = 0;
                    iconCompat.f244g = bArr2.length;
                    break;
                }
            case 2:
            case 4:
            case 6:
                String str3 = new String(iconCompat.f241d, Charset.forName("UTF-16"));
                iconCompat.f240c = str3;
                if (iconCompat.f239b == 2 && iconCompat.k == null) {
                    iconCompat.k = str3.split(":", -1)[0];
                    break;
                }
                break;
            case 3:
                iconCompat.f240c = iconCompat.f241d;
                break;
        }
        return iconCompat;
    }

    public static void write(IconCompat iconCompat, a aVar) {
        Objects.requireNonNull(aVar);
        iconCompat.j = iconCompat.i.name();
        switch (iconCompat.f239b) {
            case -1:
                iconCompat.f242e = (Parcelable) iconCompat.f240c;
                break;
            case 1:
            case 5:
                iconCompat.f242e = (Parcelable) iconCompat.f240c;
                break;
            case 2:
                iconCompat.f241d = ((String) iconCompat.f240c).getBytes(Charset.forName("UTF-16"));
                break;
            case 3:
                iconCompat.f241d = (byte[]) iconCompat.f240c;
                break;
            case 4:
            case 6:
                iconCompat.f241d = iconCompat.f240c.toString().getBytes(Charset.forName("UTF-16"));
                break;
        }
        int i = iconCompat.f239b;
        if (-1 != i) {
            aVar.p(1);
            aVar.t(i);
        }
        byte[] bArr = iconCompat.f241d;
        if (bArr != null) {
            aVar.p(2);
            aVar.r(bArr);
        }
        Parcelable parcelable = iconCompat.f242e;
        if (parcelable != null) {
            aVar.p(3);
            aVar.u(parcelable);
        }
        int i2 = iconCompat.f243f;
        if (i2 != 0) {
            aVar.p(4);
            aVar.t(i2);
        }
        int i3 = iconCompat.f244g;
        if (i3 != 0) {
            aVar.p(5);
            aVar.t(i3);
        }
        ColorStateList colorStateList = iconCompat.f245h;
        if (colorStateList != null) {
            aVar.p(6);
            aVar.u(colorStateList);
        }
        String str = iconCompat.j;
        if (str != null) {
            aVar.p(7);
            aVar.v(str);
        }
        String str2 = iconCompat.k;
        if (str2 != null) {
            aVar.p(8);
            aVar.v(str2);
        }
    }
}