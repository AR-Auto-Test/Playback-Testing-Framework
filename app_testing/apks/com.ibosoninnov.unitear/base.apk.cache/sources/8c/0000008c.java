package androidx.core.graphics.drawable;

import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapShader;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Shader;
import android.graphics.drawable.Icon;
import android.net.Uri;
import android.os.Build;
import android.os.Parcelable;
import android.text.TextUtils;
import android.util.Log;
import androidx.versionedparcelable.CustomVersionedParcelable;
import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.lang.reflect.InvocationTargetException;

/* loaded from: classes.dex */
public class IconCompat extends CustomVersionedParcelable {

    /* renamed from: a  reason: collision with root package name */
    public static final PorterDuff.Mode f238a = PorterDuff.Mode.SRC_IN;

    /* renamed from: b  reason: collision with root package name */
    public int f239b;

    /* renamed from: c  reason: collision with root package name */
    public Object f240c;

    /* renamed from: d  reason: collision with root package name */
    public byte[] f241d;

    /* renamed from: e  reason: collision with root package name */
    public Parcelable f242e;

    /* renamed from: f  reason: collision with root package name */
    public int f243f;

    /* renamed from: g  reason: collision with root package name */
    public int f244g;

    /* renamed from: h  reason: collision with root package name */
    public ColorStateList f245h;
    public PorterDuff.Mode i;
    public String j;
    public String k;

    public IconCompat() {
        this.f239b = -1;
        this.f241d = null;
        this.f242e = null;
        this.f243f = 0;
        this.f244g = 0;
        this.f245h = null;
        this.i = f238a;
        this.j = null;
    }

    public static Bitmap a(Bitmap bitmap, boolean z) {
        int min = (int) (Math.min(bitmap.getWidth(), bitmap.getHeight()) * 0.6666667f);
        Bitmap createBitmap = Bitmap.createBitmap(min, min, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(createBitmap);
        Paint paint = new Paint(3);
        float f2 = min;
        float f3 = 0.5f * f2;
        float f4 = 0.9166667f * f3;
        if (z) {
            float f5 = 0.010416667f * f2;
            paint.setColor(0);
            paint.setShadowLayer(f5, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f2 * 0.020833334f, 1023410176);
            canvas.drawCircle(f3, f3, f4, paint);
            paint.setShadowLayer(f5, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 503316480);
            canvas.drawCircle(f3, f3, f4, paint);
            paint.clearShadowLayer();
        }
        paint.setColor(-16777216);
        Shader.TileMode tileMode = Shader.TileMode.CLAMP;
        BitmapShader bitmapShader = new BitmapShader(bitmap, tileMode, tileMode);
        Matrix matrix = new Matrix();
        matrix.setTranslate((-(bitmap.getWidth() - min)) / 2, (-(bitmap.getHeight() - min)) / 2);
        bitmapShader.setLocalMatrix(matrix);
        paint.setShader(bitmapShader);
        canvas.drawCircle(f3, f3, f4, paint);
        canvas.setBitmap(null);
        return createBitmap;
    }

    public static IconCompat b(Resources resources, String str, int i) {
        if (i != 0) {
            IconCompat iconCompat = new IconCompat(2);
            iconCompat.f243f = i;
            iconCompat.f240c = str;
            iconCompat.k = str;
            return iconCompat;
        }
        throw new IllegalArgumentException("Drawable resource ID must not be 0");
    }

    public int c() {
        int i = this.f239b;
        if (i != -1) {
            if (i == 2) {
                return this.f243f;
            }
            throw new IllegalStateException("called getResId() on " + this);
        }
        int i2 = Build.VERSION.SDK_INT;
        Icon icon = (Icon) this.f240c;
        if (i2 >= 28) {
            return icon.getResId();
        }
        try {
            return ((Integer) icon.getClass().getMethod("getResId", new Class[0]).invoke(icon, new Object[0])).intValue();
        } catch (IllegalAccessException e2) {
            Log.e("IconCompat", "Unable to get icon resource", e2);
            return 0;
        } catch (NoSuchMethodException e3) {
            Log.e("IconCompat", "Unable to get icon resource", e3);
            return 0;
        } catch (InvocationTargetException e4) {
            Log.e("IconCompat", "Unable to get icon resource", e4);
            return 0;
        }
    }

    public Uri d() {
        int i = this.f239b;
        if (i != -1) {
            if (i != 4 && i != 6) {
                throw new IllegalStateException("called getUri() on " + this);
            }
            return Uri.parse((String) this.f240c);
        }
        int i2 = Build.VERSION.SDK_INT;
        Icon icon = (Icon) this.f240c;
        if (i2 >= 28) {
            return icon.getUri();
        }
        try {
            return (Uri) icon.getClass().getMethod("getUri", new Class[0]).invoke(icon, new Object[0]);
        } catch (IllegalAccessException e2) {
            Log.e("IconCompat", "Unable to get icon uri", e2);
            return null;
        } catch (NoSuchMethodException e3) {
            Log.e("IconCompat", "Unable to get icon uri", e3);
            return null;
        } catch (InvocationTargetException e4) {
            Log.e("IconCompat", "Unable to get icon uri", e4);
            return null;
        }
    }

    @Deprecated
    public Icon e() {
        Icon createWithBitmap;
        int i = this.f239b;
        String str = null;
        switch (i) {
            case -1:
                return (Icon) this.f240c;
            case 0:
            default:
                throw new IllegalArgumentException("Unknown type");
            case 1:
                createWithBitmap = Icon.createWithBitmap((Bitmap) this.f240c);
                break;
            case 2:
                if (i == -1) {
                    int i2 = Build.VERSION.SDK_INT;
                    Icon icon = (Icon) this.f240c;
                    if (i2 >= 28) {
                        str = icon.getResPackage();
                    } else {
                        try {
                            str = (String) icon.getClass().getMethod("getResPackage", new Class[0]).invoke(icon, new Object[0]);
                        } catch (IllegalAccessException e2) {
                            Log.e("IconCompat", "Unable to get icon package", e2);
                        } catch (NoSuchMethodException e3) {
                            Log.e("IconCompat", "Unable to get icon package", e3);
                        } catch (InvocationTargetException e4) {
                            Log.e("IconCompat", "Unable to get icon package", e4);
                        }
                    }
                } else if (i == 2) {
                    if (TextUtils.isEmpty(this.k)) {
                        str = ((String) this.f240c).split(":", -1)[0];
                    } else {
                        str = this.k;
                    }
                } else {
                    throw new IllegalStateException("called getResPackage() on " + this);
                }
                createWithBitmap = Icon.createWithResource(str, this.f243f);
                break;
            case 3:
                createWithBitmap = Icon.createWithData((byte[]) this.f240c, this.f243f, this.f244g);
                break;
            case 4:
                createWithBitmap = Icon.createWithContentUri((String) this.f240c);
                break;
            case 5:
                if (Build.VERSION.SDK_INT >= 26) {
                    createWithBitmap = Icon.createWithAdaptiveBitmap((Bitmap) this.f240c);
                    break;
                } else {
                    createWithBitmap = Icon.createWithBitmap(a((Bitmap) this.f240c, false));
                    break;
                }
            case 6:
                if (Build.VERSION.SDK_INT >= 30) {
                    createWithBitmap = Icon.createWithAdaptiveBitmapContentUri(d());
                    break;
                } else {
                    StringBuilder x = a.x("Context is required to resolve the file uri of the icon: ");
                    x.append(d());
                    throw new IllegalArgumentException(x.toString());
                }
        }
        ColorStateList colorStateList = this.f245h;
        if (colorStateList != null) {
            createWithBitmap.setTintList(colorStateList);
        }
        PorterDuff.Mode mode = this.i;
        if (mode != f238a) {
            createWithBitmap.setTintMode(mode);
            return createWithBitmap;
        }
        return createWithBitmap;
    }

    public String toString() {
        String str;
        if (this.f239b == -1) {
            return String.valueOf(this.f240c);
        }
        StringBuilder sb = new StringBuilder("Icon(typ=");
        switch (this.f239b) {
            case 1:
                str = "BITMAP";
                break;
            case 2:
                str = "RESOURCE";
                break;
            case 3:
                str = "DATA";
                break;
            case 4:
                str = "URI";
                break;
            case 5:
                str = "BITMAP_MASKABLE";
                break;
            case 6:
                str = "URI_MASKABLE";
                break;
            default:
                str = "UNKNOWN";
                break;
        }
        sb.append(str);
        switch (this.f239b) {
            case 1:
            case 5:
                sb.append(" size=");
                sb.append(((Bitmap) this.f240c).getWidth());
                sb.append("x");
                sb.append(((Bitmap) this.f240c).getHeight());
                break;
            case 2:
                sb.append(" pkg=");
                sb.append(this.k);
                sb.append(" id=");
                sb.append(String.format("0x%08x", Integer.valueOf(c())));
                break;
            case 3:
                sb.append(" len=");
                sb.append(this.f243f);
                if (this.f244g != 0) {
                    sb.append(" off=");
                    sb.append(this.f244g);
                    break;
                }
                break;
            case 4:
            case 6:
                sb.append(" uri=");
                sb.append(this.f240c);
                break;
        }
        if (this.f245h != null) {
            sb.append(" tint=");
            sb.append(this.f245h);
        }
        if (this.i != f238a) {
            sb.append(" mode=");
            sb.append(this.i);
        }
        sb.append(")");
        return sb.toString();
    }

    public IconCompat(int i) {
        this.f239b = -1;
        this.f241d = null;
        this.f242e = null;
        this.f243f = 0;
        this.f244g = 0;
        this.f245h = null;
        this.i = f238a;
        this.j = null;
        this.f239b = i;
    }
}