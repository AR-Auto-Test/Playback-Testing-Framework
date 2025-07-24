package c.e.b;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.Texture;
import com.ibosoninnov.unitear.Player360Activity;
import com.ibosoninnov.unitear.R;
import java.io.IOException;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;

/* compiled from: Player360Activity.java */
/* loaded from: classes2.dex */
public class se implements f.e {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Player360Activity f5233a;

    public se(Player360Activity player360Activity) {
        this.f5233a = player360Activity;
    }

    @Override // f.e
    public void a(f.d dVar, f.b0 b0Var) {
        if (b0Var.B()) {
            Log.d(this.f5233a.s, "Bitmap loaded");
            final Bitmap decodeStream = BitmapFactory.decodeStream(b0Var.f5730h.B());
            this.f5233a.runOnUiThread(new Runnable() { // from class: c.e.b.lb
                @Override // java.lang.Runnable
                public final void run() {
                    int i;
                    se seVar = se.this;
                    Bitmap bitmap = decodeStream;
                    final Player360Activity player360Activity = seVar.f5233a;
                    Log.d(player360Activity.s, "createImageViewerSceneform");
                    if (bitmap == null) {
                        Log.e(player360Activity.s, "createImageViewerSceneform null bitmap");
                        player360Activity.finish();
                        return;
                    }
                    final float height = bitmap.getHeight();
                    final float width = bitmap.getWidth();
                    if (height > 4000.0f || width > 4000.0f) {
                        int width2 = bitmap.getWidth();
                        int height2 = bitmap.getHeight();
                        int i2 = 3999;
                        if (width2 > 4000 || height2 > 4000) {
                            float f2 = width2 / height2;
                            float f3 = 3999;
                            if (width2 > height2) {
                                i = (int) (f3 / f2);
                            } else {
                                i = 3999;
                                i2 = (int) (f3 * f2);
                            }
                        } else {
                            i2 = width2;
                            i = height2;
                        }
                        Matrix matrix = new Matrix();
                        matrix.postScale(i2 / width2, i / height2);
                        Bitmap createBitmap = Bitmap.createBitmap(i2, i, Bitmap.Config.ARGB_8888);
                        try {
                            new Canvas(createBitmap).drawBitmap(bitmap, matrix, null);
                            bitmap = createBitmap;
                        } catch (Exception e2) {
                            StringBuilder x = c.b.a.a.a.x("Error resizing image: ");
                            x.append(e2.getMessage());
                            Log.e("ImageUtils", x.toString());
                            bitmap = null;
                        }
                    }
                    Node node = new Node();
                    player360Activity.D = node;
                    node.setParent(player360Activity.u.getScene());
                    player360Activity.D.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
                    Texture.builder().setSource(bitmap).build().thenAccept(new Consumer() { // from class: c.e.b.qb
                        @Override // java.util.function.Consumer
                        public final void accept(Object obj) {
                            final Player360Activity player360Activity2 = Player360Activity.this;
                            float f4 = width;
                            float f5 = height;
                            final Texture texture = (Texture) obj;
                            String str = player360Activity2.s;
                            Log.d(str, "createImageViewerSceneform texture loaded " + f4 + " x " + f5);
                            Material.builder().setSource(player360Activity2, R.raw.sceneform_opaque_textured_material_doublesided).build().thenAccept(new Consumer() { // from class: c.e.b.ob
                                @Override // java.util.function.Consumer
                                public final void accept(Object obj2) {
                                    Player360Activity player360Activity3 = Player360Activity.this;
                                    Texture texture2 = texture;
                                    Material material = (Material) obj2;
                                    Objects.requireNonNull(player360Activity3);
                                    material.setTexture("texture", texture2);
                                    ModelRenderable makeSphere = ShapeFactory.makeSphere(0.2f, Vector3.zero(), material);
                                    makeSphere.setShadowCaster(false);
                                    makeSphere.setShadowReceiver(false);
                                    player360Activity3.D.setRenderable(makeSphere);
                                    c.b.a.a.a.C(0.06f, 0.06f, 0.06f, player360Activity3.D);
                                    player360Activity3.x.setVisibility(8);
                                }
                            }).exceptionally(new Function() { // from class: c.e.b.pb
                                @Override // java.util.function.Function
                                public final Object apply(Object obj2) {
                                    Throwable th = (Throwable) obj2;
                                    Log.e(Player360Activity.this.s, "Unable to load  renderable");
                                    return null;
                                }
                            });
                        }
                    }).exceptionally(new Function() { // from class: c.e.b.kb
                        @Override // java.util.function.Function
                        public final Object apply(Object obj) {
                            Throwable th = (Throwable) obj;
                            Log.e(Player360Activity.this.s, "Unable to load  texture");
                            return null;
                        }
                    });
                }
            });
            return;
        }
        Log.e(this.f5233a.s, "load bitmap response unsucessfull");
        this.f5233a.finish();
    }

    @Override // f.e
    public void b(f.d dVar, IOException iOException) {
        String str = this.f5233a.s;
        StringBuilder x = c.b.a.a.a.x("load bitmap ");
        x.append(iOException.toString());
        Log.e(str, x.toString());
        this.f5233a.finish();
    }
}