import sys
sys.path.insert(0, "/home/claude/w/tools")
from wdoc import Doc, ptext, q, fa_dash

d = Doc("/home/claude/w/ch2.docx")


def replace_in_runs(p, old, new):
    for t in p.iter(q("t")):
        if t.text and old in t.text:
            t.text = t.text.replace(old, fa_dash(new))
            return
    raise KeyError(old[:50])


p = d.find("ترتیب مطالب به این صورت است")
replace_in_runs(p, "سازوکار توجه و معماری‌های ترکیبی چندوجهی، اختصاص دارد",
                "سازوکار توجه، معماری‌های ترکیبی چندوجهی و روش‌های آموزش مدل‌های خودبازگشتی، اختصاص دارد")

h_t = d.find("۲-۴-۴- معماری‌های ترکیبی و ادغام چندوجهی")
body_t = d.find("ترکیب شبکهٔ کانولوشنی و شبکهٔ بازگشتی سابقهٔ طولانی")
anchor = d.find("افق پیش‌بینی تنها یک گام است")

texts = [
    ("h", "۲-۴-۵- آموزش مدل‌های خودبازگشتی و هزینهٔ طیفی"),
    ("p", "مدل این پژوهش در اجرای زنده خودبازگشتی است؛ یعنی خروجی هر گام آن بخشی از ورودی گام‌های بعد را می‌سازد. رایج‌ترین روش آموزش چنین مدل‌هایی «آموزش با معلم» است که ویلیامز و زیپسر (Williams & Zipser, 1989) برای شبکه‌های بازگشتی معرفی کردند: در آموزش، به جای خروجی مدل، مقدار واقعی گام قبل به ورودی داده می‌شود. این روش آموزش را ساده و پایدار می‌کند، اما بین آموزش و اجرا تفاوت ایجاد می‌کند؛ مدل در آموزش همیشه ورودی درست می‌بیند و در اجرا ورودی‌ای که از خطاهای خودش ساخته شده است. رانزاتو و همکاران (Ranzato et al., 2016) این مشکل را سوگیری مواجهه (exposure bias) نامیدند و نشان دادند که خطاهای کوچک در دنباله‌های تولیدشده روی هم جمع می‌شوند."),
    ("p", "برای کاهش این مشکل چند راه پیشنهاد شده است. ونکاترامان و همکاران (Venkatraman et al., 2015) در پیش‌بینی چندگامی سری زمانی، پیش‌بینی‌های خود مدل را در طول آموزش به آن برمی‌گرداندند تا مدل یاد بگیرد از حالت‌هایی که خودش ایجاد می‌کند به مسیر درست برگردد. بنجیو و همکاران (Bengio et al., 2015) در روش نمونه‌برداری زمان‌بندی‌شده، به‌تدریج و به‌صورت تصادفی بخشی از ورودی‌های واقعی را با خروجی مدل جایگزین کردند. لمب و همکاران (Lamb et al., 2016) هم در روش Professor Forcing از یک شبکهٔ متمایزکننده استفاده کردند تا رفتار شبکه در حالت آموزش با معلم و حالت تولید آزاد به هم نزدیک شود. در حوزه‌ای نزدیک‌تر به این پژوهش، مارتینز و همکاران (Martinez et al., 2017) در پیش‌بینی حرکت بدن انسان با شبکه‌های بازگشتی، خروجی‌های خود مدل را در آموزش به آن برگرداندند تا از جمع شدن خطا جلوگیری کنند. آن‌ها همچنین نشان دادند که یک خط پایهٔ بسیار ساده، یعنی تکرار آخرین وضعیت، از بسیاری از مدل‌های منتشرشده بهتر عمل می‌کند؛ نکته‌ای که لزوم مقایسه با خط پایه‌های ساده را نشان می‌دهد."),
    ("p", "مشکل دوم به تابع هزینه مربوط است. وقتی آینده قطعی نیست، مدلی که با میانگین مربعات خطا آموزش ببیند میانگین حالت‌های ممکن را پیش‌بینی می‌کند. ماتیو و همکاران (Mathieu et al., 2016) این اثر را در پیش‌بینی فریم‌های ویدیو نشان دادند: فریم‌هایی که با این هزینه پیش‌بینی می‌شوند تار هستند، چون میانگین چند آیندهٔ ممکن یک تصویر تار است. برای سیگنال‌های نوسانی مثل حرکت ناشی از گام، اگر فاز نوسان از روی ورودی معلوم نباشد، میانگین حالت‌های ممکن نزدیک به صفر است و مدل به سمت خروجی کوچک میل می‌کند. در سنتز صدا، که با مشکل مشابهی روبه‌رو است، از هزینه‌هایی استفاده می‌شود که سیگنال تولیدشده را از روی طیفش با سیگنال واقعی مقایسه می‌کنند. انگل و همکاران (Engel et al., 2020) در روش DDSP از یک هزینهٔ طیفی چندمقیاسی، شامل اختلاف دامنه و لگاریتم دامنهٔ طیف در چند اندازهٔ پنجره، استفاده کردند. چون دامنهٔ طیف به فاز وابسته نیست، چنین هزینه‌ای مدل را برای تولید محتوای فرکانسی درست تشویق می‌کند، بدون اینکه هم‌ترازی دقیق زمانی را لازم داشته باشد."),
    ("p", "راه کامل‌تر، استفاده از مدل‌های مولد است که به جای یک خروجی قطعی، توزیع خروجی‌های ممکن را یاد می‌گیرند. در انیمیشن کاراکتر، راه دیگری هم برای حرکت‌های چرخه‌ای به کار رفته است: در شبکهٔ PFNN (Holden et al., 2017) فاز چرخهٔ راه رفتن به‌صورت صریح به شبکه داده می‌شود و وزن‌های شبکه تابعی از این فاز هستند. در این پژوهش، به دلیل محدودیت زمان اجرا و حجم داده، مدل قطعی باقی مانده و از ترکیب آموزش چندگامی و یک هزینهٔ طیفی استفاده شده است (بخش ۳-۶-۲). مدل‌های مولد و ورودی فاز در فصل پنجم به‌عنوان پیشنهاد مطرح می‌شوند."),
]
for kind, txt in texts:
    anchor = d.insert_after(anchor, h_t if kind == "h" else body_t, txt)

p = d.find("باید روی بازه‌های حرکت عمدی")
d.insert_after(p, body_t, "برای مقایسهٔ حرکت تولیدشده با حرکت واقعی، علاوه بر معیارهای نرمی، معیارهای طیفی هم به کار می‌روند. طیف توان یک سیگنال معمولاً با روش ولچ (Welch, 1967) تخمین زده می‌شود. در این روش سیگنال به بخش‌های هم‌پوشان تقسیم می‌شود، طیف هر بخش پس از ضرب در یک پنجره محاسبه می‌شود و از این طیف‌ها میانگین گرفته می‌شود تا واریانس تخمین کم شود. برای مقایسهٔ دو طیف، در پردازش گفتار از فاصلهٔ طیفی لگاریتمی استفاده می‌شود (Gray & Markel, 1976) که ریشهٔ میانگین مربع اختلاف لگاریتم دو طیف، بر حسب دسی‌بل، است. این معیار به فاز سیگنال حساس نیست و اختلاف در بسامدهای کم‌توان را هم به اندازهٔ بسامدهای پرتوان نشان می‌دهد؛ بنابراین برای مقایسهٔ حرکتی که قرار نیست نقطه‌به‌نقطه با حرکت واقعی منطبق باشد مناسب است.")

for ref in [
    "Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *Advances in Neural Information Processing Systems, 28*, 1171–1179.",
    "Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP: Differentiable digital signal processing. *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*.",
    "Gray, A. H., & Markel, J. D. (1976). Distance measures for speech processing. *IEEE Transactions on Acoustics, Speech, and Signal Processing, 24*(5), 380–391.",
    "Lamb, A., Goyal, A., Zhang, Y., Zhang, S., Courville, A., & Bengio, Y. (2016). Professor forcing: A new algorithm for training recurrent networks. *Advances in Neural Information Processing Systems, 29*, 4601–4609.",
    "Martinez, J., Black, M. J., & Romero, J. (2017). On human motion prediction using recurrent neural networks. *Proceedings of CVPR 2017*, 2891–2900.",
    "Mathieu, M., Couprie, C., & LeCun, Y. (2016). Deep multi-scale video prediction beyond mean square error. *Proceedings of the 4th International Conference on Learning Representations (ICLR 2016)*.",
    "Ranzato, M., Chopra, S., Auli, M., & Zaremba, W. (2016). Sequence level training with recurrent neural networks. *Proceedings of the 4th International Conference on Learning Representations (ICLR 2016)*.",
    "Venkatraman, A., Hebert, M., & Bagnell, J. A. (2015). Improving multi-step prediction of learned time series models. *Proceedings of the AAAI Conference on Artificial Intelligence, 29*(1), 3024–3030.",
    "Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics, 15*(2), 70–73.",
    "Williams, R. J., & Zipser, D. (1989). A learning algorithm for continually running fully recurrent neural networks. *Neural Computation, 1*(2), 270–280.",
]:
    d.add_reference("منابع فصل دوم", ref)

d.save("/home/claude/w/out/ch2.docx")
print("ok")
